#!/bin/bash

dataset=$1
language=$2
timestamp=$(date +%s)

DATASET=${dataset} LANGUAGE=${language} \
sbatch  --job-name="train-tokenizer-${dataset}-${language}" \
        --output="./logs/train_tokenizer_${dataset}_${language}_${timestamp}.out" \
        scripts/train_tokenizer.euler