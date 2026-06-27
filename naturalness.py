"""
naturalness.py — the "keep it natural" oracle, with two interchangeable backends.

Both backends expose the SAME interface so the optimizers don't care which one
they're using:

    ll_soft(batch)  : (B,20,L) soft one-hot -> (B,)  differentiable log-likelihood
    ll_hard(seq)    : str -> float                    log-likelihood of a real sequence
    embed(seqs)     : list[str] -> (n,d) numpy        pooled embedding (used by BO)

Higher log-likelihood = more natural. Pick the backend with --nat_model.

ESM-2  (esm)   : masked PLM, bidirectional. Pseudo-log-likelihood in ONE forward
                 pass:  sum_i  <P_i, log_softmax(logits_i)>.  Generic "looks like
                 a protein".

ProGen2 (progen): autoregressive LM. TRUE log-likelihood via the chain rule:
                 sum_i log P(x_i | x_<i). If you fine-tune ProGen2 on a protein
                 family (e.g. Cas9-like), this measures "natural FOR THIS FAMILY",
                 which is a much better leash than generic ESM.

The whole sequence is wrapped as  `1 <residues> 2`  (ProGen2's control tokens),
and we restrict the softmax to the 20 amino-acid columns (token ids below).
"""

import numpy as np
import torch
import torch.nn as nn

from utils.encoding import AA_ALPHABET, AA_TO_IDX, sequence_to_onehot


# ── backend 1: ESM-2 pseudo-log-likelihood ────────────────────────────────────

class EsmNaturalness:
    def __init__(self, esm_model: str, device: str):
        from oracles.esm2_backbone import ESM2SoftBackbone
        from oracles.esm2_loglik import ESM2LogLikOracle
        backbone = ESM2SoftBackbone(model_name=esm_model, freeze=True)
        self.oracle = ESM2LogLikOracle(backbone).to(device).eval()
        self.device = device

    def ll_soft(self, batch):
        return self.oracle(batch)

    def ll_hard(self, seq):
        x = sequence_to_onehot(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.oracle(x)[0].item()

    def embed(self, seqs):
        h = self.oracle.backbone.hard_forward(list(seqs))   # (n, L+2, d)
        return h[:, 1:-1, :].mean(dim=1).cpu().numpy()


# ── backend 2: ProGen2 autoregressive log-likelihood ──────────────────────────

# Canonical ProGen2 vocabulary (same across all ProGen2 checkpoints):
#   id 3 = '1' (start control)   id 4 = '2' (end control)   ids 5..29 = amino acids
_START_ID, _END_ID = 3, 4
_PROGEN_AA_ID = {  # our 20 canonical AAs -> ProGen2 token id
    "A": 5,  "C": 7,  "D": 8,  "E": 9,  "F": 10, "G": 11, "H": 12, "I": 13,
    "K": 14, "L": 15, "M": 16, "N": 17, "P": 19, "Q": 20, "R": 21, "S": 22,
    "T": 23, "V": 25, "W": 26, "Y": 28,
}


class Progen2Naturalness:
    def __init__(self, ckpt: str, device: str):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True)
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        # token ids for our 20 AAs, in AA_ALPHABET order (so column j == AA_ALPHABET[j])
        self.aa_ids = torch.tensor([_PROGEN_AA_ID[a] for a in AA_ALPHABET],
                                   dtype=torch.long, device=device)
        self.wte = self.model.transformer.wte   # token embedding table

    def _embeds_from_soft(self, batch):
        """(B,20,L) soft one-hot -> (B, L+2, d) input embeddings  ( 1 <residues> 2 )."""
        B, A, L = batch.shape
        W = self.wte.weight                              # (V, d)
        aa_emb = W[self.aa_ids]                          # (20, d)
        soft = torch.einsum("bal,ad->bld", batch, aa_emb)        # (B, L, d)
        start = W[_START_ID].view(1, 1, -1).expand(B, 1, -1)
        end   = W[_END_ID].view(1, 1, -1).expand(B, 1, -1)
        return torch.cat([start, soft, end], dim=1)      # (B, L+2, d)

    def ll_soft(self, batch):
        B, A, L = batch.shape
        emb = self._embeds_from_soft(batch)
        logits = self.model(inputs_embeds=emb).logits     # (B, L+2, V)
        # position i predicts token i+1; positions 0..L-1 predict the L residues
        res_logits = logits[:, 0:L, :]                    # (B, L, V)
        log_probs = torch.log_softmax(res_logits[:, :, self.aa_ids], dim=-1)  # (B, L, 20)
        P = batch.permute(0, 2, 1)                        # (B, L, 20)
        return (P * log_probs).sum(dim=(1, 2))            # (B,)

    def _ids(self, seqs):
        return torch.tensor(
            [[_START_ID] + [_PROGEN_AA_ID[c] for c in s] + [_END_ID] for s in seqs],
            dtype=torch.long, device=self.device,
        )

    def ll_hard(self, seq):
        ids = self._ids([seq])
        with torch.no_grad():
            logits = self.model(input_ids=ids).logits[0]  # (L+2, V)
        L = len(seq)
        log_probs = torch.log_softmax(logits[0:L][:, self.aa_ids], dim=-1)  # (L, 20)
        tgt = torch.tensor([AA_TO_IDX[c] for c in seq], device=self.device)
        return log_probs[torch.arange(L, device=self.device), tgt].sum().item()

    def embed(self, seqs):
        ids = self._ids(list(seqs))
        with torch.no_grad():
            out = self.model.transformer(input_ids=ids)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return h[:, 1:-1, :].mean(dim=1).float().cpu().numpy()


# ── factory ───────────────────────────────────────────────────────────────────

def build_naturalness(kind: str, device: str, esm_model: str, progen_ckpt: str):
    if kind == "esm":
        print(f"[naturalness] ESM-2 pseudo-log-likelihood ({esm_model})")
        return EsmNaturalness(esm_model, device)
    if kind == "progen":
        print(f"[naturalness] ProGen2 autoregressive log-likelihood ({progen_ckpt})")
        return Progen2Naturalness(progen_ckpt, device)
    raise ValueError(f"--nat_model must be 'esm' or 'progen', got {kind!r}")
