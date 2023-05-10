#!/bin/bash

model_arch=$1
dataset=$2
language=$3
seed=$4
timestamp=$(date +%s)

MODEL=${model_arch} DATASET=${dataset} LANGUAGE=${language} SEED=${seed}\
sbatch  --job-name="lm-train_${model_arch}_${language}" \
        --output="./logs/train_${model_arch}_${language}_${timestamp}.out" \
        scripts/train.euler