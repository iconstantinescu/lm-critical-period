DATA_DIR=$1
SEED=$2
EXTRA_FLAGS=$3

python ./src/learn/run_clm.py \
    --model_type gpt2 \
    --tokenizer_name "${DATA_DIR}/gpt2_tokenizer" \
    --cache_dir "${DATA_DIR}/cache" \
    --train_file "${DATA_DIR}/raw/train.txt" \
    --validation_file "${DATA_DIR}/raw/validation.txt" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --block_size 512 \
    --do_train \
    --do_eval \
    --output_dir "./checkpoints/gpt2-clm-${SEED}" \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --fp16 \
    --seed ${SEED} \
    ${EXTRA_FLAGS}