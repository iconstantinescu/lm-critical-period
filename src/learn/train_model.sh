#!/bin/bash

export WANDB__SERVICE_WAIT=300
export WANDB_PROJECT=$PROJECT

DATE=$(date +%d%m)

DATA_DIR="data/${DATASET}/${LANG1}"
MODEL_NAME="${MODEL}-${LANG1}-${SEED}-${DATE}"

extra_flags=""

if [ $DO_TEST = true ]
then
  extra_flags="${extra_flags} --gradient_accumulation_steps 1 --logging_steps 1 --max_eval_samples 120 --max_train_samples 120"
fi

if [ ! -z "${CHECKPOINT}" ]
then
  MODEL_NAME=${CHECKPOINT}
  extra_flags="${extra_flags} --resume_from_checkpoint checkpoints/${CHECKPOINT}"
fi

case $MODEL in
  gpt2)
    application="train_gpt2.sh"
    ;;

  roberta)
    application="train_roberta.sh"
    ;;

  *)
    echo -n "unknown model name"
    exit 1
    ;;
esac


DATA_DIR=${DATA_DIR} MODEL_NAME=${MODEL_NAME} \
bash "./src/learn/${application}" ${extra_flags}