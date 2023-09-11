#!/bin/bash

export WANDB__SERVICE_WAIT=300
export WANDB_PROJECT=$PROJECT

DATE=$(date +%d%m)

DATA_DIR="data/${DATASET}/${LANG1}"
MODEL_NAME="${MODEL}-${CONFIG}-${LANG1}${LANG2}-${MODE}-${SEED}-${DATE}"

extra_flags=$(<"./src/learn/configs/${MODEL}_${CONFIG}.txt")

# Add extra flags for test training
if [ "$DO_TEST" = true ]
then
  echo "Running experiment on test mode"
  extra_flags="${extra_flags} --gradient_accumulation_steps 1 --logging_steps 1 --max_eval_samples 120 --max_train_samples 120"
fi

# Check if we resume training from checkpoint
if [ ! -z "${CHECKPOINT}" ]
then
  if [ ! -z "${LANG2}" ] && [ "$MODE" = "sequential" ]
  then
    extra_flags="${extra_flags}  --model_name_or_path checkpoints/${CHECKPOINT}"


    if [ "$USE_EWC" = true ]
    then
      echo "Train model with elastic weight consolidation"
      extra_flags="${extra_flags} --do-ewc"
    fi
  else
    MODEL_NAME=${CHECKPOINT}
    extra_flags="${extra_flags} --resume_from_checkpoint checkpoints/${CHECKPOINT}"
  fi
fi

# Select which model type to train
case "$MODEL" in
  gpt2)
    application="train_gpt2.sh"
    ;;

  roberta)
    application="train_roberta.sh"
    ;;

  *)
    echo -n "unknown model name"
    return 1
    ;;
esac

# Train default tokenizer if it has not been trained before
if [ ! -z "${TOKENIZER}" ]
then
  echo "Using custom tokenizer ${TOKENIZER}"
  extra_flags="${extra_flags} --tokenizer_name data/${DATASET}/${TOKENIZER}"

elif [ -d "${DATA_DIR}/${MODEL}_tokenizer" ];
then
  echo "Default tokenizer is already trained."
else
	echo "Did not find trained tokenizer. Training from scratch."
	python3 ./src/learn/train_tokenizer.py ${MODEL} ${DATASET} ${LANG1}
fi


if [ ! -z "${SWEEP_ID}" ]
then
  # Hyperparameter sweep
  export MODEL_NAME="${MODEL}-${LANG1}-${SEED}-${DATE}-${IDX}"
  export DATA_DIR=${DATA_DIR}

  wandb agent --count 5 ${SWEEP_ID}

else
  if [ "$MODE" = "sequential" ]
  then
    if [ ! -z "${LANG2}" ]
    then
      # Train second language for sequential mode
      export MODEL_NAME="${MODEL_NAME}-2"
      export DATA_DIR="data/${DATASET}/${LANG2}"

      bash ./src/learn/${application} ${extra_flags}

    elif [ ! -z "${LANG1}" ]
    then
      # Train first language for sequential mode
      export MODEL_NAME="${MODEL_NAME}-1"
      export DATA_DIR=${DATA_DIR}

      bash ./src/learn/${application} ${extra_flags}

    fi

  elif [ "$MODE" = "interleaved" ];
  then
    # Train on interleaved dataset
    export MODEL_NAME="${MODEL_NAME}"
    export DATA_DIR="data/${DATASET}/${LANG1}_${LANG2}"

    bash ./src/learn/${application} ${extra_flags}
  else
    echo "Invalid MODE selected: ${MODE}."
  fi
fi


