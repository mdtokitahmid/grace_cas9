#!/bin/bash
# run.sh — submit an immunogenicity-optimization job to Slurm.
#
# These jobs MUST run on a compute node (the login node kills ESM forward passes).
#
# Usage:
#   bash run.sh grace   results/cas9_grace.json   [extra args...]
#   bash run.sh pcgrad  results/cas9_pcgrad.json  [extra args...]
#   bash run.sh bo      results/cas9_bo.json      [extra args...]
#
# Edit SEQ_FILE / ORACLE / hyperparameters below for your protein.

set -eo pipefail
BASE=/scratch/gpfs/MONA/Toki/GRACE/immuno_opt
cd "$BASE"
mkdir -p logs results

METHOD="${1:-grace}"
OUT="${2:-results/${METHOD}.json}"
shift 2 2>/dev/null || shift $# 2>/dev/null
EXTRA="$*"

SEQ_FILE="cas9.fasta"                              # <- your protein
ORACLE="stub"                                       # <- path to your 9-mer oracle, or 'stub'
NAT_MODEL="progen"                                  # naturalness: 'esm' or 'progen'
ESM_MODEL="facebook/esm2_t30_150M_UR50D"            # used when NAT_MODEL=esm (35M is faster)
PROGEN_CKPT="hugohrban/progen2-small"               # <- your Cas9-finetuned ProGen2 checkpoint

sbatch \
    --job-name="immuno_${METHOD}" \
    --output="logs/immuno_${METHOD}_%j.out" \
    --error="logs/immuno_${METHOD}_%j.err" \
    --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=01:30:00 \
    --mail-type=FAIL --mail-user=mt3204@princeton.edu \
    --wrap="
        set -eo pipefail
        module purge
        module load anaconda3/2024.2
        source \"\$(conda info --base)/etc/profile.d/conda.sh\"
        conda activate tftrain
        cd ${BASE}
        export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
        echo 'Method: ${METHOD} | Node: '\$(hostname)
        python -u optimize.py \
            --seq_file ${SEQ_FILE} \
            --oracle ${ORACLE} \
            --method ${METHOD} \
            --theta 0.5 --nat_drop 15.0 --ham_lambda 0.01 \
            --steps 300 --K 4 --lr 1e-2 \
            --nat_model ${NAT_MODEL} \
            --esm_model ${ESM_MODEL} \
            --progen_ckpt ${PROGEN_CKPT} \
            --out ${OUT} \
            ${EXTRA}
    "
echo "Submitted immuno_${METHOD}  ->  ${OUT}"
echo "Logs: logs/immuno_${METHOD}_<jobid>.out / .err"