#!/bin/bash

MODEL_TYPE="roberta"
LANG="de"

for model_dir in ./checkpoints/${MODEL_TYPE}-config2*-${LANG}*/; do
    echo model_dir
    for checkpoint in model_dir/checkpoint-*/; do
      echo checkpoint
    done
    echo "\n"
done