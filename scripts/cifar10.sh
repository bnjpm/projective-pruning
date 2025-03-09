# --------
# CIFAR-10 Resnet56
# --------

# pretraining
python main.py --seed 1 --mode pretrain --dataset cifar10 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model resnet56
python main.py --seed 1 --mode pretrain --dataset cifar10 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model vgg19
python main.py --seed 1 --mode pretrain --dataset cifar10 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model resnext50
python main.py --seed 1 --mode pretrain --dataset cifar10 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model googlenet
python main.py --seed 1 --mode pretrain --dataset cifar10 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model densenet121
python main.py --seed 1 --mode pretrain --dataset cifar10 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model inceptionv4
python main.py --seed 1 --mode pretrain --dataset cifar10 --total-epochs 200 --lr 0.1 --lr-decay-milestones 120,160,180 --model mobilenetv2

# CIFAR-10
python main.py --mode pretrain --dataset cifar10 --model resnet56 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180
python main.py --mode pretrain --dataset cifar10 --model vgg19 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180

# CIFAR-100
python main.py --mode pretrain --dataset cifar100 --model resnet56 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180
python main.py --mode pretrain --dataset cifar100 --model vgg19 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180
python main.py --mode pretrain --dataset cifar100 --model densenet121 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180
python main.py --mode pretrain --dataset cifar100 --model googlenet --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180
python main.py --mode pretrain --dataset cifar100 --model mobilenetv2 --lr 0.05 --total-epochs 200 --lr-decay-milestones 120,150,180
python main.py --mode pretrain --dataset cifar100 --model resnext50 --lr 0.1 --total-epochs 200 --lr-decay-milestones 120,150,180

# 2x speed up
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

# 3x speed up
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

# VGG19 on CIFAR-10

# 2x speed up
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

# 3x speed up
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
