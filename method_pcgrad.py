"""
method_pcgrad.py — PCGrad baseline (Project Conflicting Gradients).

Same soft-sequence setup as GRACE, but the constraint is handled differently.
We treat the problem as two "tasks":

  task 1: lower immunogenicity        -> descent grad g_imm
  task 2: fix naturalness violation   -> descent grad of relu(floor - LL)
          (zero whenever naturalness is already fine)

PCGrad: if two task gradients conflict (negative dot product), project each one
to remove the conflicting component, then add them. A small Hamming penalty is
added on top. This is the standard multi-objective baseline to compare against
GRACE's null-space projection.
"""

import random

import torch

from common import (
    ProteinSeqProp, sequence_to_onehot, expected_hamming,
    grad_vec, set_grad,
)


def pcgrad_project(g_tasks):
    """Project away pairwise-conflicting components, return the summed direction."""
    proj = [g.clone() for g in g_tasks]
    order = list(range(len(g_tasks)))
    random.shuffle(order)
    for i in order:
        for j in order:
            if i == j:
                continue
            dot = torch.dot(proj[i], g_tasks[j])
            if dot < 0:   # conflict -> remove the offending component
                proj[i] = proj[i] - (dot / (g_tasks[j].norm().square() + 1e-12)) * g_tasks[j]
    return sum(proj)


def optimize(wt_seq, immuno, naturalness, evaluate, wt_record, cfg):
    device = cfg["device"]
    L = len(wt_seq)

    seqprop = ProteinSeqProp(L, init_sequence=wt_seq, init_gamma=3.0).to(device)
    params  = list(seqprop.parameters())
    opt     = torch.optim.Adam(params, lr=cfg["lr"])
    X_orig  = sequence_to_onehot(wt_seq).to(device)

    floor = wt_record["naturalness_ll"] - cfg["nat_drop"]

    candidates = {}
    history = []

    for step in range(cfg["steps"]):
        opt.zero_grad()
        samples, P = seqprop.st_sample(cfg["K"])
        batch = torch.stack(samples, dim=0)

        L_imm = immuno.hinge_soft(batch).mean()
        ll    = naturalness.ll_soft(batch).mean()
        viol  = torch.relu(floor - ll)                  # naturalness violation (>=0)

        g_imm  = grad_vec(L_imm, params)
        g_viol = grad_vec(viol,  params)                # zero if not violated

        combined = pcgrad_project([g_imm, g_viol])
        g_ham = grad_vec(cfg["ham_lambda"] * expected_hamming(P, X_orig), params, retain=False)

        set_grad(params, combined + g_ham)
        opt.step()

        with torch.no_grad():
            seq = seqprop.decode("argmax")
        rec = evaluate(seq)
        candidates[seq] = rec
        history.append({"step": step, "immuno_hinge": rec["immuno_hinge"],
                        "ll_drop": rec["ll_drop"], "n_mut": rec["n_mut"]})

        print(f"[pcgrad] step {step:4d}/{cfg['steps']} | immuno_hinge {rec['immuno_hinge']:9.4f} "
              f"(soft {L_imm.item():9.4f}) | epitopes {rec['n_epitopes']:3d} "
              f"| ll {rec['naturalness_ll']:9.3f} ll_drop {rec['ll_drop']:7.3f} "
              f"| n_mut {rec['n_mut']:3d} | viol {viol.item():.3f}", flush=True)

    with torch.no_grad():
        for _ in range(cfg["n_final"]):
            seq = seqprop.decode("sample")
            if seq not in candidates:
                candidates[seq] = evaluate(seq)

    return list(candidates.values()), history
