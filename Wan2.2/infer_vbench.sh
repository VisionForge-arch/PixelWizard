#!/bin/bash

CKPT=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_240p/checkpoint_model_000100/model.pt
OUT_ROOT=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/eval_vbench/240p_5s

for SEED in 0 1 2 3 4
do
    echo "Running with base_seed=${SEED}"

    torchrun --nproc_per_node=8 \
        generate_multiple2.py \
        --size=448*256 \
        --save_file=${OUT_ROOT}/seed_${SEED}/pt \
        --wan_ckpt=${CKPT} \
        --use_ema \
        --dit_fsdp \
        --t5_fsdp \
        --ulysses_size 8 \
        --base_seed ${SEED}

    echo "Finished seed ${SEED}"
done