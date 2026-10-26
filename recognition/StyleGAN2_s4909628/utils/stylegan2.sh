#!/bin/bash
#SBATCH --job-name=stylegan2_adni
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Submit from this directory after creating logs/.
set -euo pipefail

# Submit from the directory containing train.py.
cd "$SLURM_SUBMIT_DIR"

# First argument: total target epochs. Default: 20.
target_epochs="${1:-20}"

if [ "$#" -gt 0 ]; then
    shift
fi

conda run --no-capture-output -n torch \
    python -u train.py \
    --device cuda \
    --epochs "$target_epochs" \
    --batch-size 8 \
    --num-workers 4 \
    "$@"