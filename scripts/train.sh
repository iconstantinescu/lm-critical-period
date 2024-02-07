#!/bin/bash

model="gpt2"
lang1="en"
dataset="unified_clean"
training_mode="sequential"
seed=0
project_name="critical-period"
resume=false
do_sweep=false
do_test=false
use_ewc=false
timestamp=$(date +%s)

Help()
{
   # Display Help
   echo "Script to run model training."
   echo
   echo "Syntax: train.sh [-n|1|2|m|d|t|s|c|f|T|p|w]"
   echo "options:"
   echo "n     Model name (gpt2 or roberta) to train from scratch. Default: gpt2"
   echo "c     The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
   echo "r     Resume training from the checkpoint instead of starting a new one"
   echo "1     First language to train on. Default: en"
   echo "2     Second language to train on"
   echo "m     Training mode: sequential or interleaved. Default: sequential"
   echo "d     Dataset to use. Default: unified_clean"
   echo "t     Path to custom tokenizer"
   echo "s     Random seed number. Default: 0"
   echo "f     Config file for model (extra flags)"
   echo "E     Use elastic weight consolidation. Default: false"
   echo "T     Run in test/debug mode (fewer samples). Default: false"
   echo "p     Project name for wandb logging. Default: critical-period"
   echo "w     Wandb sweep id for hyperparameter tuning (sweep must be started already)"

   echo
}

while getopts "n:c:1:2:m:d:t:s:f:p:w:rETh" option; do
  case $option in
    n)
      model="$OPTARG"
      ;;
    c)
      checkpoint="$OPTARG"
      ;;
    r)
      resume=true
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
    t)
      tokenizer="$OPTARG"
      ;;
    s)
      seed="$OPTARG"
      ;;
    f)
      config_file="$OPTARG"
      ;;
    p)
      project_name="$OPTARG"
      ;;
    w)
      sweep_id="$OPTARG"
      do_sweep=true
      ;;
    E)
      use_ewc=true
      ;;
    T)
      do_test=true
      project_name="test"
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
echo "Checkpoint: $checkpoint"
echo "Resume training: $resume"
echo "Language 1: $lang1"
echo "Language 2: $lang2"
echo "Training mode: $training_mode"
echo "Dataset: $dataset"
echo "Custom tokenizer: $tokenizer"
echo "Seed: $seed"
echo "Configuration file: $config_file"
echo "Do test: $do_test"
echo "Do sweep: $do_sweep"
echo "Use ewc: $use_ewc"
echo "Project name: $project_name"

if [ $do_sweep = true ] ;
then
  echo 'Doing hyperparameter sweep'

  # Run sweep agents in parallel
  for i in 1 2 3 4
  do
    MODEL=${model} DATASET=${dataset} TOKENIZER=${tokenizer} LANG1=${lang1} LANG2=${lang2} MODE=${training_mode} SEED=${seed} \
    PROJECT=${project_name} DO_TEST=${do_test} SWEEP_ID=${sweep_id} IDX=${i} \
    sbatch  --job-name="sweep-${model}-${lang1}${lang2}-${training_mode}" \
            --output="./logs/sweeps/sweep_${model}_${lang1}${lang2}_${training_mode}_${seed}_${timestamp}_${i}.out" \
            scripts/train.euler
  done


else
  echo 'Doing normal training'
  MODEL=${model} CHECKPOINT=${checkpoint} RESUME=${resume} DATASET=${dataset} TOKENIZER=${tokenizer} LANG1=${lang1} LANG2=${lang2} MODE=${training_mode} \
  SEED=${seed} CONFIG=${config_file} USE_EWC=${use_ewc} DO_TEST=${do_test} PROJECT=${project_name} \
  sbatch  --job-name="lm-train-${model}-${lang1}${lang2}-${training_mode}" \
          --output="./logs/trainings/train_${model}_${config_file}_${lang1}${lang2}_${training_mode}_${seed}_${timestamp}.out" \
          scripts/train.euler
fi


echo
