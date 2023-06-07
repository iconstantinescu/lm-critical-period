#!/bin/bash

module purge
module load eth_proxy gcc/8.2.0 python_gpu/3.9.9
source /cluster/work/cotterell/iconstantine/lm-critical-period/venv/bin/activate

model=$1
shift

project_name="critical-period"
training_mode="sequential"


while getopts "1:2:m:p:" option; do
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
    p)
      project_name="$OPTARG"
      ;;
    *)
      echo "Usage: $0 [-f file_name] [-d directory_name]"
      exit 1
      ;;
  esac
done


wandb sweep --project ${project_name} --name "${model}-${lang1}${lang2}-${training_mode}"  ./src/learn/sweep_config_${model}.yaml