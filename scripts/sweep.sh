#!/bin/bash

module purge
module load eth_proxy gcc/8.2.0 python_gpu/3.9.9
source /cluster/work/cotterell/iconstantine/lm-critical-period/venv/bin/activate


config_name="gpt2"
project_name="critical-period"
training_mode="sequential"


while getopts "n:1:2:m:p:" option; do
  case $option in
    n)
      config_name="$OPTARG"
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
    p)
      project_name="$OPTARG"
      ;;
    *)
      echo "Usage: $0 [-n config_name] [-1 first language] [-2 second language] [-m training_mode] [-p project_name]"
      exit 1
      ;;
  esac
done


wandb sweep --project ${project_name} --name "${config_name}-${lang1}${lang2}-${training_mode}"  ./src/learn/configs/sweep_${config_name}.yaml