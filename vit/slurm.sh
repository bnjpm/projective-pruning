#!/bin/bash
#SBATCH --job-name=projprune_experiment
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output="logs/%J-%x.out"
#SBATCH --error="logs/%J-%x.err"
#SBATCH --gres=gpu:1

nvidia-smi

echo "Start: $(date)"
source .venv/bin/activate

python3 main.py "$1" "$2" "$3"

deactivate
echo "Finish: $(date)"

