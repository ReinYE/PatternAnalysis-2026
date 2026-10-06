#!/bin/bash
#SBATCH --job-name=dcgan_adni
#SBATCH --partition=comp3710
#SBATCH --account=comp3710

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

conda activate torch
python -u train.py

