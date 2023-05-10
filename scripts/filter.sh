#!/bin/bash

dataset=$1
language=$2
timestamp=$(date +%s)

DATASET=${dataset} LANGUAGE=${language} \
sbatch --output="./logs/filter_${dataset}_${language}_${timestamp}.out" scripts/filter.euler