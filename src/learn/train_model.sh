#!/bin/bash

export WANDB__SERVICE_WAIT=300
export WANDB_PROJECT=$PROJECT

DATE=$(date +%d%m)

DATA_DIR="data/${DATASET}/${LANG1}"
MODEL_NAME="${MODEL}-${LANG1}-${SEED}-${DATE}"

extra_flags=""

# Add extra flags for test training
if [ $DO_TEST = true ]
then
  extra_flags="${extra_flags} --gradient_accumulation_steps 1 --logging_steps 1 --max_eval_samples 120 --max_train_samples 120"
fi

# Check if we resume training from checkpoint
if [ ! -z "${CHECKPOINT}" ]
then
  MODEL_NAME=${CHECKPOINT}
  extra_flags="${extra_flags} --resume_from_checkpoint checkpoints/${CHECKPOINT}"
fi

# Select which model type to train
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

# Train tokenizer if it has not been trained before
if [ -d "${DATA_DIR}/${MODEL}_tokenizer" ];
then
  echo "Tokenizer is already trained."
else
	echo "Did not find trained tokenizer. Training from scratch."
	python3 ./src/learn/train_tokenizer.py ${MODEL} ${DATASET} ${LANG1}
fi

export DATA_DIR=${DATA_DIR}

if [ ! -z "${SWEEP_ID}" ]
then
  # Hyperparameter sweep
  export MODEL_NAME="${MODEL_NAME}-${IDX}"
  wandb agent --count 10 ${SWEEP_ID}
else
  # Normal training
  export MODEL_NAME=${MODEL_NAME}
  bash ./src/learn/${application} ${extra_flags}
fi


