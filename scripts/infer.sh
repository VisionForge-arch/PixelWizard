#!/bin/bash
# Unified LR → SR pipeline (2K output)
# Usage: bash scripts/infer.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --standalone --nproc_per_node=8 \
    generate.py \
    --ckpt_dir <YOUR_PATH>/Wan2.2-TI2V-5B \
    --sr_ckpt <YOUR_PATH>/sr_2k_checkpoint/model.pt \
    --prompt_file <YOUR_PATH>/prompts.txt \
    --save_dir <YOUR_PATH>/outputs/sr_2k \
    --lr_size 448*256 \
    --lr_steps 50 \
    --sr_size 2560*1440 \
    --sr_steps 4 \
    --sr_shift 5.5 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8
