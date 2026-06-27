"""
method_grace.py — GRACE: null-space projected gradient descent.

We keep a soft sequence (SeqProp) and, each step:

  g_imm = gradient that LOWERS immunogenicity        (we descend this)
  g_ll  = gradient that RAISES naturalness (ESM LL)

We project g_imm so it is orthogonal to g_ll:

      d = g_imm - (g_imm . g_ll_hat) * g_ll_hat

Descending d lowers immunogenicity WITHOUT changing the naturalness to first
order — that's the whole point: we move along the "free" direction that the
naturalness constraint allows. If naturalness ever falls below the floor
(LL < LL_wt - nat_drop), we add a one-sided pull-back along g_ll to bring it
back up. A small Hamming penalty discourages pointless mutations.
"""

import torch

from common import (
    ProteinSeqProp, sequence_to_onehot, expected_hamming,
    grad_vec, set_grad,
)


def optimize(wt_seq, immuno, naturalness, evaluate, wt_record, cfg):
    device = cfg["device"]
    L = len(wt_seq)

    seqprop = ProteinSeqProp(L, init_sequence=wt_seq, init_gamma=3.0).to(device)
    params  = list(seqprop.parameters())
    opt     = torch.optim.Adam(params, lr=cfg["lr"])
    X_orig  = sequence_to_onehot(wt_seq).to(device)

    floor = wt_record["naturalness_ll"] - cfg["nat_drop"]   # naturalness must stay >= floor

    candidates = {}   # seq -> record  (dedup)
    history = []

    for step in range(cfg["steps"]):
        opt.zero_grad()
        samples, P = seqprop.st_sample(cfg["K"])
        batch = torch.stack(samples, dim=0)                 # (K, 20, L)

        L_imm = immuno.hinge_soft(batch).mean()             # want to minimize
        ll    = naturalness.ll_soft(batch).mean()           # want >= floor

        g_imm = grad_vec(L_imm, params)                     # descent dir for immunogenicity
        g_ll  = grad_vec(ll,    params)                     # ascent dir for naturalness
        g_hat = g_ll / (g_ll.norm() + 1e-12)

        # project out the component of g_imm that would change naturalness
        d = g_imm - (g_imm @ g_hat) * g_hat

        # one-sided floor: if naturalness dropped too far, push it back up
        if ll.item() < floor:
            d = d - cfg["lam_floor"] * g_ll

        # discourage unnecessary mutations
        g_ham = grad_vec(cfg["ham_lambda"] * expected_hamming(P, X_orig), params, retain=False)

        set_grad(params, d + g_ham)
        opt.step()

        # decode + evaluate the current best-guess sequence
        with torch.no_grad():
            seq = seqprop.decode("argmax")
        rec = evaluate(seq)
        candidates[seq] = rec
        history.append({"step": step, "immuno_hinge": rec["immuno_hinge"],
                        "ll_drop": rec["ll_drop"], "n_mut": rec["n_mut"]})

        # projection_ratio = how much of the immuno gradient survived the naturalness
        # projection. ~1 -> objectives compatible; ~0 -> tightly coupled / stuck.
        proj_ratio = (d.norm() / (g_imm.norm() + 1e-12)).item()
        print(f"[grace] step {step:4d}/{cfg['steps']} | immuno_hinge {rec['immuno_hinge']:9.4f} "
              f"(soft {L_imm.item():9.4f}) | epitopes {rec['n_epitopes']:3d} "
              f"| ll {rec['naturalness_ll']:9.3f} ll_drop {rec['ll_drop']:7.3f} "
              f"| n_mut {rec['n_mut']:3d} | proj {proj_ratio:.3f}"
              + ("  [floor!]" if ll.item() < floor else ""), flush=True)

    # final: sample a few sequences from the converged distribution
    with torch.no_grad():
        for _ in range(cfg["n_final"]):
            seq = seqprop.decode("sample")
            if seq not in candidates:
                candidates[seq] = evaluate(seq)

    return list(candidates.values()), history
