#!/bin/bash
#SBATCH --job-name=prune
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output="jobs/%J-%x.out"
#SBATCH --error="jobs/%J-%x.out"
#SBATCH --gres=gpu:1

echo "SLURM WORKLOAD START: $(date)"
start=$(date +%s)
nvidia-smi
source .venv/bin/activate
python3 main.py cifar10_vgg16
python3 main.py cifar100_vgg16
python3 main.py imagenet_vgg16
python3 main.py imagenet_alexnet
deactivate
end=$(date +%s)
diff=$((end - start))
echo "TIME TAKEN: $(date -ud "@$diff" +'%T')"
echo "SLURM WORKLOAD FINISH: $(date)"
