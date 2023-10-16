#!/bin/bash

EXTRA_FLAGS="$@"
echo $EXTRA_FLAGS

# RUN_APPLICATION="python"
RUN_APPLICATION="torchrun --nproc_per_node 4"

${RUN_APPLICATION} ./src/learn/run_clm.py \
    --model_name_or_path "./checkpoints/${MODEL_NAME}" \
    --validation_file "${DATA_DIR}/raw/validation.txt" \
    --cache_dir "${DATA_DIR}/cache" \
    --run_name "eval_${MODEL_NAME}" \
    --seed ${SEED} \
    --report_to wandb \
    --output_dir "./checkpoints/${MODEL_NAME}" \
    --overwrite_output_dir \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 128 \
    --block_size 512 \
    --do_eval \
    --low_cpu_mem_usage \
    ${EXTRA_FLAGS}