#!/bin/bash

model="gpt2"
lang1="en"
dataset="unified_clean"
training_mode="sequential"
seed=42
project_name="critical-period"
do_sweep=false
do_test=false
timestamp=$(date +%s)

Help()
{
   # Display Help
   echo "Script to run model training."
   echo
   echo "Syntax: train.sh [-n|1|2|m|d|s|c|t|w]"
   echo "options:"
   echo "n     Model name (gpt2 or roberta). Default: gpt2"
   echo "1     First language to train on. Default: en"
   echo "2     Second language to train on"
   echo "m     Training mode: sequential or interleaved. Default: sequential"
   echo "d     Dataset to use. Default: unified_clean"
   echo "s     Random seed number. Default: 42"
   echo "c     Checkpoint path to resume training"
   echo "t     Run in test/debug mode (fewer samples). Default: false"
   echo "p     Project name for wandb logging. Default: critical-period"
   echo "w     Wandb sweep id for hyperparameter tuning (sweep must be started already)"

   echo
}

while getopts "n:1:2:m:d:s:c:p:w:th" option; do
  case $option in
    n)
      model="$OPTARG"
      ;;
    1)
      lang1="$OPTARG"
      ;;
    2)
      lang2="$OPTARG"
      ;;
    m)
      training_mode="$OPTARG"
      ;;
    d)
      dataset="$OPTARG"
      ;;
    s)
      seed="$OPTARG"
      ;;
    c)
      checkpoint="$OPTARG"
      ;;
    t)
      do_test=true
      project_name="test"
      ;;
    p)
      project_name="$OPTARG"
      ;;
    w)
      sweep_id="$OPTARG"
      do_sweep=true
      ;;
    h)
      Help
      exit
      ;;
    *)
      echo "Usage: $0 [-n model_name] [-d dataset] [-1 first language]"
      exit 1
      ;;
  esac
done

echo
echo "Model: $model"
echo "Language 1: $lang1"
echo "Language 2: $lang2"
echo "Training mode: $training_mode"
echo "Dataset: $dataset"
echo "Seed: $seed"
echo "Checkpoint: $checkpoint"
echo "Do test: $do_test"
echo "Do sweep: $do_sweep"
echo "Project name: $project_name"

if [ $do_sweep = true ] ;
then
  echo 'Doing hyperparameter sweep'

  # Run sweep agents in parallel
  for i in 1 2 3 4
  do
    MODEL=${model} DATASET=${dataset} LANG1=${lang1} LANG2=${lang2} MODE=${training_mode} SEED=${seed} \
    PROJECT=${project_name} DO_TEST=${do_test} SWEEP_ID=${sweep_id} IDX=${i} \
    sbatch  --job-name="sweep-${model}-${lang1}${lang2}-${training_mode}" \
            --output="./logs/sweeps/sweep_${model}_${lang1}${lang2}_${training_mode}_${seed}_${timestamp}_${i}.out" \
            scripts/train.euler
  done


else
  echo 'Doing normal training'
  MODEL=${model} DATASET=${dataset} LANG1=${lang1} LANG2=${lang2} MODE=${training_mode} SEED=${seed} \
  PROJECT=${project_name} CHECKPOINT=${checkpoint} DO_TEST=${do_test} \
  sbatch  --job-name="lm-train-${model}-${lang1}${lang2}-${training_mode}" \
          --output="./logs/trainings/train_${model}_${lang1}${lang2}_${training_mode}_${seed}_${timestamp}.out" \
          scripts/train.euler
fi


echo
