# ----------
# ModelNet40
# ----------

# pretraining
python main.py --seed 1 --mode pretrain --dataset modelnet40 --batch-size 32 --total-epochs 250 --lr 0.1 --lr-decay-milestones 100,160,200 ---model dgcnn
python main.py --seed 1 --mode pretrain --dataset modelnet40 --batch-size 32 --total-epochs 250 --lr 0.1 --lr-decay-milestones 100,160,200 ---model pointnet

# pruning + finetuning

# 2x speed up
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method random
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method l2
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method fpgm
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method obdc
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method lamp
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method slim
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method group_norm
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method group_sl
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method proj
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 2 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method proj_sl

# 3x speed up
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method random
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method l2
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method fpgm
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method obdc
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method lamp
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method slim
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method group_norm
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method group_sl
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method proj
python main.py --seed 1 --mode prune --dataset modelnet40 --restore run/modelnet40/pretrain/modelnet40_dgcnn.pth --finetune --global-pruning --reg 1e-5 --speed-up 3 -soft-rank 0.5 --lr 0.01 --lr-decay-milestones 50,80 --batch-size 32 --model dgcnn --method proj_sl
