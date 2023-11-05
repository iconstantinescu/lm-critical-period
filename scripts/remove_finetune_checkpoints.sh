#!/bin/bash

#SBATCH --time=3:00:00

for model_dir in ./checkpoints/*; do
    echo $model_dir
    for task_dir in $model_dir/finetune/*; do
      CMD="rm -r $task_dir/checkpoint*"
      echo $CMD
      eval $CMD
      wait
    done
    echo -e "\n"
done