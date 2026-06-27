"""
method_bo.py — constrained Bayesian optimization baseline.

Discrete search (no gradients). We keep a pool of evaluated sequences and, each
round, fit two Gaussian-process surrogates on their ESM embeddings:

  GP_immuno : embedding -> immunogenicity hinge   (the objective, minimize)
  GP_nat    : embedding -> naturalness LL          (the constraint)

We then generate many random mutants of the current feasible bests, score them
with a constrained acquisition:

  acq = ExpectedImprovement(lower immuno) * P(naturalness >= floor)

evaluate the top few with the real oracles, add them to the pool, and repeat.
sklearn GPs keep this short and dependency-light (no botorch needed).
"""

import random

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from common import AA_ALPHABET


def random_mutant(seq, k_min, k_max):
    s = list(seq)
    k = random.randint(k_min, k_max)
    for i in random.sample(range(len(s)), k):
        s[i] = random.choice([a for a in AA_ALPHABET if a != s[i]])
    return "".join(s)


def _fit_gp(X, y):
    kernel = ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6)
    gp.fit(X, y)
    return gp


def optimize(wt_seq, immuno, naturalness, evaluate, wt_record, cfg):
    floor = wt_record["naturalness_ll"] - cfg["nat_drop"]
    k_min, k_max = 1, cfg["max_mut"]

    # ── initial design: WT + random mutants ──────────────────────────────────
    pool = {wt_seq: wt_record}
    for _ in range(cfg["bo_n_init"]):
        seq = random_mutant(wt_seq, k_min, k_max)
        if seq not in pool:
            pool[seq] = evaluate(seq)

    history = []
    budget = cfg["steps"]                 # reuse --steps as the BO evaluation budget
    n_eval = len(pool)

    while n_eval < budget:
        seqs = list(pool.keys())
        recs = [pool[s] for s in seqs]
        X = naturalness.embed(seqs)                                  # (n, d)
        y_imm = np.array([r["immuno_hinge"]   for r in recs])
        y_ll  = np.array([r["naturalness_ll"] for r in recs])

        gp_imm = _fit_gp(X, y_imm)
        gp_nat = _fit_gp(X, y_ll)

        # best immunogenicity among CURRENTLY feasible points (incumbent)
        feasible = [r["immuno_hinge"] for r in recs if r["naturalness_ll"] >= floor]
        best = min(feasible) if feasible else min(y_imm)

        # ── propose candidates: random mutants of feasible bests ─────────────
        seeds = sorted(recs, key=lambda r: r["immuno_hinge"])[:5]
        cand = set()
        while len(cand) < cfg["bo_pool"]:
            seed = random.choice(seeds)["sequence"]
            c = random_mutant(seed, k_min, k_max)
            if c not in pool:
                cand.add(c)
        cand = list(cand)

        Xc = naturalness.embed(cand)
        mu_i, sd_i = gp_imm.predict(Xc, return_std=True)
        mu_n, sd_n = gp_nat.predict(Xc, return_std=True)

        # Expected Improvement for MINIMIZING immunogenicity
        z = (best - mu_i) / (sd_i + 1e-9)
        ei = (best - mu_i) * norm.cdf(z) + sd_i * norm.pdf(z)
        ei = np.clip(ei, 0, None)
        # probability the naturalness constraint holds
        p_feas = norm.cdf((mu_n - floor) / (sd_n + 1e-9))
        acq = ei * p_feas

        # evaluate the top-k acquisition candidates with the real oracles
        k = min(cfg["bo_batch"], budget - n_eval)
        for idx in np.argsort(-acq)[:k]:
            seq = cand[idx]
            pool[seq] = evaluate(seq)
            n_eval += 1

        cur_best = min(r["immuno_hinge"] for r in pool.values()
                       if r["naturalness_ll"] >= floor) if \
            any(r["naturalness_ll"] >= floor for r in pool.values()) else float("nan")
        history.append({"n_eval": n_eval, "best_feasible_immuno": cur_best})
        print(f"[bo] evals {n_eval:4d}/{budget} | best feasible immuno_hinge {cur_best:9.4f} "
              f"(WT {wt_record['immuno_hinge']:.4f})", flush=True)

    return list(pool.values()), history
