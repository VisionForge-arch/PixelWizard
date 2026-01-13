#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
CKPT=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_240p/checkpoint_model_000100/model.pt
OUT_ROOT=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/eval_vbench/240p_5s
PROMPE_FILE=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/prompt_vbench.jsonl

for SEED in 0 1 2 3 4
do
    echo "Running with base_seed=${SEED}"

    torchrun --standalone --nproc_per_node=6\
        generate_multiple2.py \
        --prompt_file=${PROMPE_FILE}$ \
        --size=448*256 \
        --save_file=${OUT_ROOT}/seed_${SEED}/pt \
        --wan_ckpt=${CKPT} \
        --use_ema \
        --dit_fsdp \
        --t5_fsdp \
        --ulysses_size 6 \
        --base_seed ${SEED}

    echo "Finished seed ${SEED}"
done