#!/bin/bash

model=$1
dataset=$2
language=$3
seed=$4
timestamp=$(date +%s)

MODEL=${model} DATASET=${dataset} LANGUAGE=${language} SEED=${seed} \
sbatch  --job-name="lm-train-${model}-${language}" \
        --output="./logs/train_${model}_${language}_${seed}_${timestamp}.out" \
        scripts/train.euler