#!/bin/bash

export WANDB__SERVICE_WAIT=300
export WANDB_PROJECT=$PROJECT

echo $PROJECT
echo $DATASET


case $MODEL in
  gpt2)
    application="train_gpt2.sh"
    ;;

  roberta)
    application="train_roberta.sh"
    ;;

  *)
    echo -n "unknown model name"
    ;;
esac