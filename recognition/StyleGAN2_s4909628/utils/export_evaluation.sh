#!/bin/bash
#SBATCH --job-name=adni_export_eval
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

output_dir="evaluation/cohort_final_${SLURM_JOB_ID}"

conda run --no-capture-output -n torch \
  python -u export_evaluation.py \
  --output-dir "$output_dir" \
  --reference-split test \
  --real-count 5000 \
  --num-images 5000 \
  --batch-size 32 \
  --warmup 10 \
  --benchmark-steps 50