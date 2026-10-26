#!/bin/bash
#SBATCH --job-name=adni_eval
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

# feature extraction
# conda run --no-capture-output -n torch \
#   python -u evaluation.py \
#   --cohort-dir evaluation/cohort_final_674672 \
#   --data-root /home/groups/comp3710/ADNI/AD_NC \
#   --inception-weights weights/weights-inception-2015-12-05-6726825d.pth \
#   --batch-size 32 \
#   --num-workers 4 \
#   --device cuda

# compute FID
# conda run --no-capture-output -n torch \
#   python -u evaluation.py fid \
#   --cache-dir evaluation/cohort_final_674672/stage_b/features \
#   --expected-count 5000 \
#   --torch-threads 4

# compute Feature Nearest-Neighbor Reference SSIM
# conda run --no-capture-output -n torch \
#   python -u evaluation.py ssim \
#   --cohort-dir evaluation/cohort_final_674672 \
#   --data-root /home/groups/comp3710/ADNI/AD_NC \
#   --device cpu \
#   --torch-threads 4

# evaluate sharpness
# conda run --no-capture-output -n torch \
#   python -u evaluation.py sharpness \
#   --phase cohort \
#   --cohort-dir evaluation/cohort_final_674672 \
#   --data-root /home/groups/comp3710/ADNI/AD_NC \
#   --protocol evaluation/cohort_final_674672/stage_b/sharpness_calibration/sharpness_protocol.json \
#   --expected-count 5000

# audit memorization
# conda run --no-capture-output -n torch \
#   python -u audit_memorization.py \
#   --cohort-dir evaluation/cohort_final_674672 \
#   --data-root /home/groups/comp3710/ADNI/AD_NC \
#   --inception-weights weights/weights-inception-2015-12-05-6726825d.pth \
#   --batch-size 32 \
#   --num-workers 4 \
#   --device cuda \
#   --top-k 5 \
#   --query-chunk 64 \
#   --review-quantile 0.01 \
#   --foreground-threshold 0.05 \
#   --examples 6 \
#   --seed 2026

# export memorization review
# conda run --no-capture-output -n torch \
#   python -u export_memorization_review.py \
#   --cohort-dir evaluation/cohort_final_674672 \
#   --data-root /home/groups/comp3710/ADNI/AD_NC \
#   --model stylegan2

# evaluate embedding diversity
# conda run --no-capture-output -n torch \
#   python -u evaluate_embedding_diversity.py \
#   --cohort-dir evaluation/cohort_final_674672 \
#   --data-root /home/groups/comp3710/ADNI/AD_NC \
#   --phase all \
#   --pca-components 50 \
#   --perplexities 30 50 \
#   --tsne-seeds 2026 \
#   --iterations 1500 \
#   --pairs 5000 \
#   --threads 4 \
#   --seed 2026

# evaluate latent
conda run --no-capture-output -n torch \
  python -u evaluate_latent.py \
  --cohort-dir evaluation/cohort_final_674672 \
  --data-root /home/groups/comp3710/ADNI/AD_NC \
  --dcgan-checkpoint runs/dcgan_adni_20261006_115201/checkpoints/final.pt \
  --stylegan2-checkpoint runs/stylegan2_adni_full_test_20261014_061632_618300Z/checkpoints/final.pt \
  --device cuda \
  --per-class 1 \
  --random-pairs 3 \
  --frames 9 \
  --steps 800 \
  --restarts 2 \
  --projection-lr 0.05 \
  --seed 202