# ---------
# CIFAR-100
# ---------

# pretraining
python main.py --seed 1 --mode pretrain --dataset cifar100 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model resnet56
python main.py --seed 1 --mode pretrain --dataset cifar100 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model vgg19
python main.py --seed 1 --mode pretrain --dataset cifar100 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model resnext50
python main.py --seed 1 --mode pretrain --dataset cifar100 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model googlenet
python main.py --seed 1 --mode pretrain --dataset cifar100 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model densenet121
python main.py --seed 1 --mode pretrain --dataset cifar100 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model inceptionv4
python main.py --seed 1 --mode pretrain --dataset cifar100 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model mobilenetv2

# pruning + finetuning

# 2x speed up
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method random
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method l2
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method fpgm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method obdc
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method lamp
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method slim
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method group_norm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method group_sl
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method proj
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 --model resnet56 --method proj_sl

# 3x speed up
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method random
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method l2
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method fpgm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method obdc
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method lamp
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method slim
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method group_norm
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method group_sl
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method proj
python main.py --seed 1 --mode prune --dataset cifar100 --restore run/cifar100/pretrain/cifar10_resnet56.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 --model resnet56 --method proj_sl
