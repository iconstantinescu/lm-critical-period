DATA_DIR=$1
SEED=$2
EXTRA_FLAGS=$3

python ./src/learn/run_mlm.py \
    --model_type roberta \
    --tokenizer_name "${DATA_DIR}/tokenizer-wiki.json" \
    --train_file "${DATA_DIR}/train.txt" \
    --validation_file "${DATA_DIR}/valid.txt" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --output_dir ./checkpoints/roberta-mlm \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --fp16 \
    --seed ${SEED} \
    ${EXTRA_FLAGS}