#!/bin/bash

MODEL=$1
MODEL_TYPE=$2
LANG=$3

for model_dir in ./checkpoints/${MODEL}-config2*-${LANG}*; do
    echo $model_dir
    for checkpoint in $model_dir/checkpoint-*; do
      CMD="./scripts/evaluate.sh blimp ${checkpoint#*checkpoints/} $MODEL_TYPE"
      echo $CMD
      eval $CMD
      sleep 300
    done
    echo "\n"
done