#!/bin/bash
# run.sh — submit an immunogenicity-optimization job to Slurm.
#
# These jobs MUST run on a compute node (the login node kills ESM forward passes).
#
# Usage:
#   bash run.sh grace   results/cas9_grace.json   [extra args...]
#   bash run.sh pcgrad  results/cas9_pcgrad.json  [extra args...]
#   bash run.sh bo      results/cas9_bo.json      [extra args...]
#   bash run.sh grace results/cas9_grace_mhcflurry.json
#
# Edit SEQ_FILE / ORACLE / hyperparameters below for your protein.

set -eo pipefail
BASE=/scratch/gpfs/MONA/ms3955/grace_cas9
cd "$BASE"
mkdir -p logs results

METHOD="${1:-grace}"
OUT="${2:-results/${METHOD}.json}"
shift 2 2>/dev/null || shift $# 2>/dev/null
EXTRA="$*"

DATA=/scratch/gpfs/MONA/ms3955/deimmunization/data
SEQ_FILE="${DATA}/individual_sequences/spcas9.fasta"
ORACLE="mhcflurry"
HLA_A="${DATA}/t10_hla_a.csv"
HLA_B="${DATA}/t10_hla_b.csv"
HLA_C="${DATA}/t10_hla_c.csv"
NAT_MODEL="progen"
PROGEN_CKPT="/scratch/gpfs/MONA/ms3955/deimmunization/progen/progen2/checkpoints/progen2-base"
ESM_MODEL="facebook/esm2_t30_150M_UR50D"

sbatch \
    --job-name="immuno_${METHOD}" \
    --output="logs/immuno_${METHOD}_%j.out" \
    --error="logs/immuno_${METHOD}_%j.out" \
    --gres=gpu:1 --mem=128G --cpus-per-task=4 --time=01:00:00 \
    --wrap="
        set -eo pipefail
        source /home/ms3955/miniconda3/etc/profile.d/conda.sh
        conda activate /scratch/gpfs/MONA/ms3955/env_immune
        cd ${BASE}
        export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
        echo 'Method: ${METHOD} | Node: '\$(hostname)
        python -u optimize.py \
            --seq_file ${SEQ_FILE} \
            --oracle ${ORACLE} \
            --oracle-max-batch 1024 \
            --hla-a ${HLA_A} --hla-b ${HLA_B} --hla-c ${HLA_C} \
            --method ${METHOD} \
            --theta 0.5 --nat_drop 0.15 --ham_lambda 0.01 \
            --steps 1000 --K 2 --lr 5e-3 \
            --nat_model ${NAT_MODEL} \
            --esm_model ${ESM_MODEL} \
            --progen_ckpt ${PROGEN_CKPT} \
            --out ${OUT} \
            ${EXTRA}
    "
echo "Submitted immuno_${METHOD}  ->  ${OUT}"
echo "Logs: logs/immuno_${METHOD}_<jobid>.out / .err"
