DATA_DIR=$1
SEED=$2
EXTRA_FLAGS=$3

DATE=$(date +%d/%m)

torchrun --nproc_per_node 8 ./src/learn/run_mlm.py \
    --model_type roberta \
    --tokenizer_name "${DATA_DIR}/roberta_tokenizer" \
    --train_file "${DATA_DIR}/raw/train.txt" \
    --validation_file "${DATA_DIR}/raw/validation.txt" \
    --cache_dir "${DATA_DIR}/cache" \
    --run_name "roberta-${DATA_DIR}-${SEED}-${DATE}" \
    --output_dir "./checkpoints/roberta-mlm-${SEED}-${DATE}" \
    --overwrite_output_dir \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 64 \
    --max_seq_length 510 \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --save_strategy "epoch" \
    --learning_rate 5e-4 \
    --mlm_probability 0.4 \
    --warmup_ratio 0.06 \
    --num_train_epochs 12 \
    --streaming \
    --low_cpu_mem_usage \
    --fp16 \
    --seed ${SEED} \
    ${EXTRA_FLAGS}