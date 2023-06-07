#!/bin/bash

model=$1
shift

dataset="unified_clean"
training_mode="sequential"
seed=42
project_name="critical-period"
do_sweep=false
do_test=false
timestamp=$(date +%s)

while getopts "1:2:m:d:s:c:w:t" option; do
  case $option in
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
    w)
      sweep_id="$OPTARG"
      do_sweep=true
      ;;
    *)
      echo "Usage: $0 [-f file_name] [-d directory_name]"
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
