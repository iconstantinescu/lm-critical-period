#!/bin/bash

model_arch=$1
dataset=$2
language=$3
timestamp=$(date +%s)

MODEL=${model_arch} DATASET=${dataset} LANGUAGE=${language} \
sbatch  --job-name="lm-train_${model_arch}_${language}" \
        --output="./logs/train_${model_arch}_${language}_${timestamp}.out" \
        scripts/train.euler