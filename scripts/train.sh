#!/bin/bash

sbatch  --job-name="lm-training-hf" \
        --output="./logs/train_hf.out" \
        scripts/train.euler