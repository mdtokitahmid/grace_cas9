# immuno_opt

Lower a protein's **immunogenicity** while keeping it **natural** (high ESM-2 log-likelihood).

Built for one protein at a time (e.g. ~1400-aa Cas9). Standalone and small. The
ESM-2 / SeqProp machinery in `utils/`, `model/` and `oracles/` is vendored from the
original `protein_grace` package, so this folder runs on its own.

## The idea

The immunogenicity oracle scores a single **9-mer** at a time. To get one number
for the whole protein we slide a 9-residue window along the sequence, score every
window, and combine them with a **hinge**:

```
immuno(seq) = sum over windows  of  relu(score_window - theta)
```

Only windows above the threshold `theta` count. A clean window contributes 0 —
and because `relu` is flat there, it contributes **zero gradient**, so the
optimizer only edits the immunogenic **hotspots**, not the whole protein. (A plain
mean would dilute a single bad epitope by ~1/1400 and barely move.)

**Naturalness** keeps the protein looking real; we hold it within `nat_drop` of the
wild-type log-likelihood. Two interchangeable backends (`--nat_model`):

* `esm` — ESM-2 **pseudo**-log-likelihood (bidirectional, one forward pass). Generic
  "looks like a protein."
* `progen` — ProGen2 **true** autoregressive log-likelihood (`Σ log P(x_i | x_<i)`).
  If you point `--progen_ckpt` at a ProGen2 **finetuned on your protein family**
  (e.g. Cas9-like), this measures "natural *for this family*" — a much better leash.
  ProGen2 natively accepts soft input embeddings, so the soft LL is exact (verified:
  `ll_soft(one-hot) == ll_hard`).

## Methods (one file each)

| file | method |
|---|---|
| `method_grace.py`  | **GRACE** — project the immunogenicity gradient orthogonal to the naturalness gradient, so naturalness is held fixed to first order (+ floor pull-back). |
| `method_pcgrad.py` | **PCGrad** — treat immunogenicity and the naturalness violation as two tasks; project away conflicting gradients. |
| `method_bo.py`     | **Constrained Bayesian optimization** — GP surrogates on ESM embeddings; acquisition = expected improvement × P(naturalness feasible). |

`optimize.py` is the central entry point that wires everything together.

## Plug in your real oracle

Your oracle is a `torch.nn.Module` with this signature (see `oracle_immuno.py`):

```
input :  (B, 20, 9)   soft one-hot 9-mers   (AAs alphabetical: ACDEFGHIKLMNPQRSTVWY)
output:  (B,)         immunogenicity score   (higher = more immunogenic)
```

Save it (`torch.save(model, "immuno.pt")`) and pass `--oracle immuno.pt`. Until
then, `--oracle stub` uses a random-weight placeholder so the pipeline runs.

## Run

```bash
conda activate tftrain
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   # della has no internet

# ESM naturalness
python optimize.py \
    --seq_file cas9.fasta --oracle stub --method grace \
    --theta 0.5 --nat_drop 5.0 --ham_lambda 0.01 --steps 300 \
    --nat_model esm --esm_model facebook/esm2_t12_35M_UR50D \
    --out results/cas9_grace.json

# ProGen2 naturalness (family-finetuned model — recommended)
python optimize.py \
    --seq_file cas9.fasta --oracle stub --method grace \
    --theta 0.5 --nat_drop 15.0 --ham_lambda 0.01 --steps 300 \
    --nat_model progen --progen_ckpt /path/to/cas9_finetuned_progen2 \
    --out results/cas9_grace_progen.json
```

Swap `--method pcgrad` or `--method bo` for the baselines (same arguments).
The `--progen_ckpt` must be a HuggingFace-format ProGen2 (loads via
`AutoModelForCausalLM(..., trust_remote_code=True)`, like the `hugohrban/progen2-*`
models). `nat_drop` is on the log-lik scale, which differs between ESM and ProGen2 —
tune it per backend.

## Key hyperparameters

| flag | meaning |
|---|---|
| `--theta`       | immunogenicity threshold; only windows above it are penalized |
| `--nat_model`   | naturalness backend: `esm` or `progen` |
| `--progen_ckpt` | ProGen2 model name/path (your family-finetuned checkpoint) |
| `--nat_drop`    | max allowed log-likelihood drop vs WT (the naturalness leash) |
| `--ham_lambda`  | mutation penalty — larger keeps fewer edits |
| `--steps`       | optimization iterations (for BO: total evaluation budget) |
| `--K`, `--lr`   | gradient methods: straight-through samples per step, learning rate |
| `--max_mut`     | BO: max substitutions per proposed candidate |

## Output

JSON with the wild-type scores and a list of **successful** sequences (naturalness
preserved AND immunogenicity improved), sorted best-first. Each record has the
sequence, `n_mut`, immunogenicity (hinge / mean / max / #epitopes), the naturalness
log-likelihood, and `ll_drop` from WT.
