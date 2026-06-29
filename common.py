"""
common.py — shared building blocks for immunogenicity optimization.

The idea
--------
We have ONE protein (e.g. Cas9, ~1400 aa). We want to:
  * lower its immunogenicity, and
  * keep it as "natural" as the wild type (high ESM-2 log-likelihood).

The immunogenicity oracle only scores a single 9-mer at a time. So to get one
number for the whole protein we slide a 9-residue window along the sequence,
score every window, and combine the per-window scores with a HINGE:

    immuno(seq) = sum_w  relu( score_w - theta )

Only windows above the threshold theta contribute. A clean window contributes 0,
and (because relu is flat there) it also contributes ZERO gradient — so the
optimizer only edits the immunogenic hotspots, not the whole sequence.

Naturalness is the ESM-2 pseudo-log-likelihood (a single forward pass over the
soft sequence). We keep it within `nat_drop` of the wild-type value.

This file holds the pieces every method (GRACE / PCGrad / BO) shares.
The ESM/SeqProp machinery in utils/, model/ and oracles/ is vendored from the
original `protein_grace` package so this folder runs standalone.
"""

import torch
import torch.nn as nn

from utils.encoding import (
    AA_ALPHABET,
    sequence_to_onehot,
    hamming_distance,
    expected_hamming,
)
from model.seqprop import ProteinSeqProp          # soft sequence distribution

WINDOW = 9   # k-mer size the immunogenicity oracle expects


# ── immunogenicity: windows + hinge ───────────────────────────────────────────

def window_scores(batch: torch.Tensor, oracle: nn.Module) -> torch.Tensor:
    """
    Score every 9-mer window of every sequence in the batch.

    batch:  (B, 20, L) soft one-hot sequences
    oracle: maps a 9-mer (B, 20, 9) -> (B,) immunogenicity score
    returns (B, num_windows) per-window scores  (differentiable)
    """
    B, A, L = batch.shape
    # unfold along the length axis into windows of size 9, stride 1
    w = batch.unfold(2, WINDOW, 1)            # (B, 20, num_windows, 9)
    nw = w.shape[2]
    w = w.permute(0, 2, 1, 3).reshape(B * nw, A, WINDOW)  # (B*nw, 20, 9)
    return oracle(w).reshape(B, nw)           # (B, num_windows)


def hinge_immuno(batch: torch.Tensor, oracle: nn.Module, theta: float) -> torch.Tensor:
    """Per-sequence hinge immunogenicity:  sum_w relu(score_w - theta).  -> (B,)"""
    s = window_scores(batch, oracle)
    return torch.relu(s - theta).sum(dim=1)


class Immuno:
    """Thin wrapper around the (swappable) 9-mer oracle + the threshold theta."""

    def __init__(self, oracle: nn.Module, theta: float, device: str):
        self.oracle = oracle.to(device).eval()
        self.theta = theta
        self.device = device

    def hinge_soft(self, batch: torch.Tensor) -> torch.Tensor:
        """Differentiable hinge objective on a soft batch (B,20,L) -> (B,)."""
        return hinge_immuno(batch, self.oracle, self.theta)

    def eval_hard(self, seq: str) -> dict:
        """Score a single decoded sequence string (no gradients)."""
        x = sequence_to_onehot(seq).unsqueeze(0).to(self.device)   # (1,20,L)
        with torch.no_grad():
            s = window_scores(x, self.oracle)[0]                   # (num_windows,)
        return {
            "hinge":   torch.relu(s - self.theta).sum().item(),
            "mean":    s.mean().item(),
            "max":     s.max().item(),
            "n_above": int((s > self.theta).sum().item()),        # # immunogenic windows
        }


# (The naturalness backends — ESM-2 and ProGen2 — live in naturalness.py.)


# ── evaluation: turn a sequence into a scored record ──────────────────────────

def make_evaluator(immuno: Immuno, naturalness, wt_seq: str):
    """
    Returns (evaluate, wt_record).
    evaluate(seq) -> dict with immunogenicity + naturalness + #mutations.
    """
    wt_im = immuno.eval_hard(wt_seq)
    wt_ll = naturalness.ll_hard(wt_seq)

    def evaluate(seq: str) -> dict:
        im = immuno.eval_hard(seq)
        ll = naturalness.ll_hard(seq)
        return {
            "sequence":       seq,
            "n_mut":          hamming_distance(wt_seq, seq),
            "immuno_hinge":   im["hinge"],
            "immuno_mean":    im["mean"],
            "immuno_max":     im["max"],
            "n_epitopes":     im["n_above"],
            "naturalness_ll": ll,
            "ll_drop":        wt_ll - ll,        # how much naturalness we gave up
        }

    wt_record = {
        "sequence": wt_seq, "n_mut": 0,
        "immuno_hinge": wt_im["hinge"], "immuno_mean": wt_im["mean"],
        "immuno_max": wt_im["max"], "n_epitopes": wt_im["n_above"],
        "naturalness_ll": wt_ll, "ll_drop": 0.0,
    }
    return evaluate, wt_record


def is_successful(rec: dict, wt: dict, nat_drop: float) -> bool:
    """Successful = naturalness preserved AND immunogenicity actually improved.
    nat_drop is in mean NLL units; ll_drop is in sum LL units."""
    natural_enough = rec["ll_drop"] <= nat_drop * len(wt["sequence"])
    improved       = rec["immuno_hinge"] < wt["immuno_hinge"]
    return natural_enough and improved


# ── tiny gradient helpers (flatten params <-> one vector) ─────────────────────

def grad_vec(loss: torch.Tensor, params, retain: bool = True) -> torch.Tensor:
    """Flattened gradient of `loss` w.r.t. all params, as a single 1-D vector."""
    grads = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])


def set_grad(params, vec: torch.Tensor):
    """Write a flat descent direction back into param.grad, ready for optimizer.step()."""
    off = 0
    for p in params:
        n = p.numel()
        p.grad = vec[off:off + n].view_as(p).clone()
        off += n
