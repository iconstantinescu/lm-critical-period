#!/bin/bash

#SBATCH --time=3:00:00

for model_dir in ./checkpoints/*; do
    echo $model_dir

    for task_dir in $model_dir/finetune/*; do

      # remove intermediate checkpoints from evaluations
      CMD1="rm -r $task_dir/checkpoint*"
      echo $CMD1
      eval $CMD1
      wait

      # remove finetuned models from evaluations
      CMD2="rm $task_dir/pytorch_model.bin"
      echo $CMD2
      eval $CMD2
      wait

    done

    for checkpoint in $model_dir/checkpoint-*; do

     # remove optimizer state from training checkpoints
      CMD3="rm $checkpoint/optimizer.pt"
      echo $CMD3
      eval $CMD3
      wait

    done

    echo -e "\n"
done