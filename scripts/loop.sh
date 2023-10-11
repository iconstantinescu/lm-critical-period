#!/bin/bash

#SBATCH --time=23:00:00

MODEL=$1
MODEL_TYPE=$2
LANG=$3

for model_dir in ./checkpoints/${MODEL}-config2*-${LANG}*; do
    echo $model_dir
    for checkpoint in $model_dir/checkpoint-*; do
      if [ "$MODEL" = gpt2 ]
      then
        echo -e "\nFixing tokenizer config for $checkpoint"
        python3 ./src/learn/fix_tokenizer.py $checkpoint
      fi

      CMD="./scripts/evaluate.sh blimp ${checkpoint#*checkpoints/} $MODEL_TYPE"
      echo $CMD
      eval $CMD
      sleep 480
    done
    echo -e "\n"
done