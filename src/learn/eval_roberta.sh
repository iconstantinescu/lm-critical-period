#!/bin/bash

EXTRA_FLAGS="$@"
echo $EXTRA_FLAGS

# RUN_APPLICATION="python"
RUN_APPLICATION="torchrun --nproc_per_node 8"

${RUN_APPLICATION} ./src/learn/run_mlm.py \
    --model_name_or_path "${CHECKPOINTS_DIR}/${MODEL_NAME}" \
    --validation_file "${DATA_DIR}/raw/validation.txt" \
    --cache_dir "${DATA_DIR}/cache" \
    --run_name "eval_${LANG}_${MODEL_NAME}" \
    --seed ${SEED} \
    --report_to wandb \
    --output_dir "${CHECKPOINTS_DIR}/${MODEL_NAME}" \
    --overwrite_output_dir \
    --per_device_eval_batch_size 8 \
    --eval_accumulation_steps 64 \
    --max_seq_length 510 \
    --do_eval \
    --low_cpu_mem_usage \
    ${EXTRA_FLAGS}