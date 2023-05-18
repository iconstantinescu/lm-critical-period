DATA_DIR=$1
SEED=$2
EXTRA_FLAGS=$3

torchrun --nproc_per_node 8 ./src/learn/run_mlm.py \
    --model_type roberta \
    --tokenizer_name "${DATA_DIR}/roberta_tokenizer" \
    --cache_dir "${DATA_DIR}/cache" \
    --train_file "${DATA_DIR}/raw/train.txt" \
    --validation_file "${DATA_DIR}/raw/validation.txt" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_seq_length 510 \
    --do_train \
    --do_eval \
    --output_dir "./checkpoints/roberta-mlm-${SEED}" \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --streaming \
    --low_cpu_mem_usage \
    --fp16 \
    --seed ${SEED} \
    ${EXTRA_FLAGS}