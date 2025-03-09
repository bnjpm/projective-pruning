#!/bin/bash
#SBATCH --job-name=gpt2_projprune
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output="logs/%J-%x.out"
#SBATCH --error="logs/%J-%x.out"
#SBATCH --gres=gpu:a100_80:1

echo "SLURM WORKLOAD START: $(date)"
start=$(date +%s)
nvidia-smi
source .venv/bin/activate
python3 main.py "$(date +%Y%m%d%H%M%S).csv"
deactivate
end=$(date +%s)
diff=$(( end - start ))
echo "TIME TAKEN: $(date -ud "@$diff" +'%T')"
echo "SLURM WORKLOAD FINISH: $(date)"
