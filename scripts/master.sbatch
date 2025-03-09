#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output="logs/%J-%x.out"
#SBATCH --error="logs/%J-%x.out"
#SBATCH --gres=gpu:1

echo "SLURM WORKLOAD START: $(date)"
start=$(date +%s)
nvidia-smi
source .venv/bin/activate

# pretrain
python main.py --seed 1 --mode pretrain --dataset cifar10 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model resnet56
python main.py --seed 1 --mode pretrain --dataset cifar10 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model vgg19
python main.py --seed 1 --mode pretrain --dataset cifar10 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model mobilenetv2
python main.py --seed 1 --mode pretrain --dataset cifar100 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model resnet56
python main.py --seed 1 --mode pretrain --dataset cifar100 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model vgg19
python main.py --seed 1 --mode pretrain --dataset cifar100 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model mobilenetv2
python main.py --seed 1 --mode pretrain --dataset modelnet40 --batch-size 32 --total-epochs 250 --lr 0.1 --lr-decay-milestones 100,160,200 --model dgcnn
python main.py --seed 1 --mode pretrain --dataset modelnet40 --batch-size 32 --total-epochs 250 --lr 0.1 --lr-decay-milestones 100,160,200 --model pointnet

# cifar10 resnet 2
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method random
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method l2
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method fpgm
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method obdc
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method lamp
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method slim
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method group_norm
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method group_sl
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method proj
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method proj_sl

# cifar10 resnet 3
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method random
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method l2
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method fpgm
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method obdc
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method lamp
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method slim
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method group_norm
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method group_sl
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method proj
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method proj_sl

# cifar10 vgg 2
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method random
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method l2
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method fpgm
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method obdc
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method lamp
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method slim
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method group_norm
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method group_sl
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method proj
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method proj_sl

# cifar10 vgg 3
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method random
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method l2
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method fpgm
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method obdc
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method lamp
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method slim
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method group_norm
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method group_sl
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method proj
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method proj_sl

# cifar10 mobilenetv2 2
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method random
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method l2
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method fpgm
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method obdc
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method lamp
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method slim
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method group_norm
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method group_sl
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method proj
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method proj_sl

# cifar10 mobilenetv2 3
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method random
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method l2
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method fpgm
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method obdc
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method lamp
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method slim
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method group_norm
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method group_sl
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method proj
python main.py --seed 1 --mode prune --dataset cifar10 --restore run/cifar10/pretrain/cifar10_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method proj_sl

# cifar100 resnet 2
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method random
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method l2
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method fpgm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method obdc
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method lamp
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method slim
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method group_norm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method group_sl
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method proj
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method proj_sl

# cifar100 resnet 3
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method random
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method l2
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method fpgm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method obdc
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method lamp
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method slim
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method group_norm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method group_sl
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method proj
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method proj_sl

# cifar100 vgg19 2
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method random
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method l2
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method fpgm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method obdc
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method lamp
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method slim
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method group_norm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method group_sl
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method proj
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model vgg19 --method proj_sl

# cifar100 vgg19 3
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method random
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method l2
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method fpgm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method obdc
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method lamp
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method slim
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method group_norm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method group_sl
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method proj
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_vgg19.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model vgg19 --method proj_sl

# cifar100 mobilenetv2 2
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method random
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method l2
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method fpgm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method obdc
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method lamp
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method slim
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method group_norm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method group_sl
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method proj
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model mobilenetv2 --method proj_sl

# cifar100 mobilenetv2 3
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method random
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method l2
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method fpgm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method obdc
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method lamp
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method slim
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method group_norm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method group_sl
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method proj
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar100_mobilenetv2.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model mobilenetv2 --method proj_sl

# imagenet resnet 4
python -m torch.distributed.launch --nproc_per_node=1 --master_port 18113 --use_env main_imagenet.py --epochs 90 --batch-size 256 --lr 0.08 --prune --cache-dataset --soft-keeping-ratio 0.5 --pretrained --target-flops 4 --global-pruning --data-path data/imagenet --model resnet50 --method random
python -m torch.distributed.launch --nproc_per_node=1 --master_port 18113 --use_env main_imagenet.py --epochs 90 --batch-size 256 --lr 0.08 --prune --cache-dataset --soft-keeping-ratio 0.5 --pretrained --target-flops 4 --global-pruning --data-path data/imagenet --model resnet50 --method l2
python -m torch.distributed.launch --nproc_per_node=1 --master_port 18113 --use_env main_imagenet.py --epochs 90 --batch-size 256 --lr 0.08 --prune --cache-dataset --soft-keeping-ratio 0.5 --pretrained --target-flops 4 --global-pruning --data-path data/imagenet --model resnet50 --method fpgm
python -m torch.distributed.launch --nproc_per_node=1 --master_port 18113 --use_env main_imagenet.py --epochs 90 --batch-size 256 --lr 0.08 --prune --cache-dataset --soft-keeping-ratio 0.5 --pretrained --target-flops 4 --global-pruning --data-path data/imagenet --model resnet50 --method obdc
python -m torch.distributed.launch --nproc_per_node=1 --master_port 18113 --use_env main_imagenet.py --epochs 90 --batch-size 256 --lr 0.08 --prune --cache-dataset --soft-keeping-ratio 0.5 --pretrained --target-flops 4 --global-pruning --data-path data/imagenet --model resnet50 --method lamp
python -m torch.distributed.launch --nproc_per_node=1 --master_port 18113 --use_env main_imagenet.py --epochs 90 --batch-size 256 --lr 0.08 --prune --cache-dataset --soft-keeping-ratio 0.5 --pretrained --target-flops 4 --global-pruning --data-path data/imagenet --model resnet50 --method slim
python -m torch.distributed.launch --nproc_per_node=1 --master_port 18113 --use_env main_imagenet.py --epochs 90 --batch-size 256 --lr 0.08 --prune --cache-dataset --soft-keeping-ratio 0.5 --pretrained --target-flops 4 --global-pruning --data-path data/imagenet --model resnet50 --method group_norm
python -m torch.distributed.launch --nproc_per_node=1 --master_port 18113 --use_env main_imagenet.py --epochs 90 --batch-size 256 --lr 0.08 --prune --cache-dataset --soft-keeping-ratio 0.5 --pretrained --target-flops 4 --global-pruning --data-path data/imagenet --model resnet50 --method group_sl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 18113 --use_env main_imagenet.py --epochs 90 --batch-size 256 --lr 0.08 --prune --cache-dataset --soft-keeping-ratio 0.5 --pretrained --target-flops 4 --global-pruning --data-path data/imagenet --model resnet50 --method proj
python -m torch.distributed.launch --nproc_per_node=1 --master_port 18113 --use_env main_imagenet.py --epochs 90 --batch-size 256 --lr 0.08 --prune --cache-dataset --soft-keeping-ratio 0.5 --pretrained --target-flops 4 --global-pruning --data-path data/imagenet --model resnet50 --method proj_sl

# modelnet pointnet 2
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method random
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method l2
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method fpgm
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method obdc
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method lamp
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method slim
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method group_norm
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method group_sl
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method proj
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method proj_sl

# modelnet pointnet 3
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method random
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method l2
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method fpgm
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method obdc
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method lamp
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method slim
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method group_norm
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method group_sl
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method proj
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_pointnet.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model pointnet --method proj_sl

deactivate
end=$(date +%s)
diff=$((end - start))
echo "TIME TAKEN: $(date -ud "@$diff" +'%T')"
echo "SLURM WORKLOAD FINISH: $(date)"
