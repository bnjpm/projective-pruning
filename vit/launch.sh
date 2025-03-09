#!/bin/bash

forms=(
    "uniform"
    "adaptive"
    "norm_analysis"
)

models=(
    "vit_b_16"
    "vit_b_32"
    "vit_l_16"
    "vit_l_32"
    "vit_h_14"
)

experiments=(
    "l1"
    "l2"
    "proj000"
    "proj500"
    "proj550"
    "proj5n50"
)

suffix="prune"

mkdir -p logs

for form in "${forms[@]}"; do
    for model in "${models[@]}"; do
        for label in "${experiments[@]}"; do
            sbatch --job-name "${label}-${model}-${form}-${suffix}" slurm.sh "$form" "$model" "$label"
            sleep 1
        done
    done
done
