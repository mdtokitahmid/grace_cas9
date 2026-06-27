"""
optimize.py — central entry point.

Takes a protein sequence + an immunogenicity oracle + hyperparameters, runs the
chosen search method, and returns the sequences that successfully LOWERED
immunogenicity while staying natural.

Example
-------
    python optimize.py \
        --seq_file cas9.fasta \
        --oracle stub \
        --method grace \
        --theta 0.5 \
        --nat_drop 5.0 \
        --ham_lambda 0.01 \
        --steps 300 \
        --esm_model facebook/esm2_t12_35M_UR50D \
        --out results/cas9_grace.json

Hyperparameters you'll usually touch:
    --theta       immunogenicity threshold; only windows scoring above it are penalized
    --nat_drop    max allowed drop in ESM log-likelihood vs wild type (the naturalness leash)
    --ham_lambda  penalty on number of mutations (bigger -> fewer edits)
    --steps       number of optimization iterations (BO: evaluation budget)
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from common import Immuno, make_evaluator, is_successful
from naturalness import build_naturalness
from oracle_immuno import load_immuno_oracle

import method_grace
import method_pcgrad
import method_bo

METHODS = {"grace": method_grace, "pcgrad": method_pcgrad, "bo": method_bo}


def read_sequence(args) -> str:
    if args.sequence:
        return args.sequence.strip().upper()
    text = Path(args.seq_file).read_text().strip().splitlines()
    # accept raw sequence or FASTA
    seq = "".join(l.strip() for l in text if not l.startswith(">"))
    return seq.upper()


def parse_args():
    p = argparse.ArgumentParser(description="Lower protein immunogenicity while staying natural.")
    # input
    p.add_argument("--sequence", default=None, help="protein sequence string")
    p.add_argument("--seq_file", default=None, help="FASTA or plain-text sequence file")
    p.add_argument("--oracle", default="stub", help="'stub' or path to a 9-mer immunogenicity nn.Module")
    p.add_argument("--method", default="grace", choices=list(METHODS))
    # core hyperparameters
    p.add_argument("--theta",      type=float, default=0.5,  help="immunogenicity threshold")
    p.add_argument("--nat_drop",   type=float, default=5.0,  help="max allowed log-lik drop vs WT")
    # naturalness backend
    p.add_argument("--nat_model",  default="esm", choices=["esm", "progen"],
                   help="naturalness oracle: ESM-2 (generic) or ProGen2 (family-finetuned)")
    p.add_argument("--progen_ckpt", default="hugohrban/progen2-small",
                   help="ProGen2 model name or path (e.g. your Cas9-finetuned checkpoint)")
    p.add_argument("--ham_lambda", type=float, default=0.01, help="mutation (Hamming) penalty")
    p.add_argument("--steps",      type=int,   default=300,  help="iterations (BO: eval budget)")
    # gradient-method knobs
    p.add_argument("--K",          type=int,   default=4,    help="straight-through samples per step")
    p.add_argument("--lr",         type=float, default=1e-2)
    p.add_argument("--lam_floor",  type=float, default=1.0,  help="naturalness floor pull-back strength")
    p.add_argument("--n_final",    type=int,   default=20,   help="sequences sampled from final distribution")
    # BO knobs
    p.add_argument("--max_mut",    type=int,   default=4,    help="BO: max mutations per proposal")
    p.add_argument("--bo_n_init",  type=int,   default=20)
    p.add_argument("--bo_pool",    type=int,   default=200,  help="BO: candidates scored per round")
    p.add_argument("--bo_batch",   type=int,   default=8,    help="BO: real evals per round")
    # misc
    p.add_argument("--esm_model",  default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out",        default="results/immuno_opt.json")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    wt_seq = read_sequence(args)
    print(f"Sequence length: {len(wt_seq)} | method: {args.method} | device: {args.device}")

    # build the two oracles
    immuno_oracle = load_immuno_oracle(args.oracle, args.device)
    immuno = Immuno(immuno_oracle, theta=args.theta, device=args.device)
    naturalness = build_naturalness(args.nat_model, args.device,
                                    esm_model=args.esm_model, progen_ckpt=args.progen_ckpt)

    # baseline (wild type) scores + a shared evaluator
    evaluate, wt_record = make_evaluator(immuno, naturalness, wt_seq)
    print(f"WT  | immuno_hinge {wt_record['immuno_hinge']:.3f} "
          f"| epitopes {wt_record['n_epitopes']} | ll {wt_record['naturalness_ll']:.3f}")

    cfg = vars(args)
    candidates, history = METHODS[args.method].optimize(
        wt_seq, immuno, naturalness, evaluate, wt_record, cfg
    )

    # keep only successful candidates, best immunogenicity first
    successful = [c for c in candidates if is_successful(c, wt_record, args.nat_drop)]
    successful.sort(key=lambda r: r["immuno_hinge"])

    result = {
        "method":     args.method,
        "config":     {k: cfg[k] for k in
                       ["theta", "nat_drop", "ham_lambda", "steps", "K", "lr",
                        "nat_model", "esm_model", "progen_ckpt", "max_mut", "seed"]},
        "wt":         wt_record,
        "n_candidates": len(candidates),
        "n_successful": len(successful),
        "successful": successful,
        "best":       successful[0] if successful else None,
        "history":    history,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"\n{len(successful)} successful sequences (of {len(candidates)} tried)")
    for r in successful[:5]:
        print(f"  immuno {r['immuno_hinge']:8.3f} (WT {wt_record['immuno_hinge']:.3f}) "
              f"| epitopes {r['n_epitopes']} | n_mut {r['n_mut']} | ll_drop {r['ll_drop']:.3f}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
