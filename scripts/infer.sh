#!/bin/bash
# PixelWizard LR anchor -> HR synthesis -> decode pipeline (2K output)
# Usage: bash scripts/infer.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --standalone --nproc_per_node=8 generate.py \
    --ckpt_dir <YOUR_PATH>/Wan2.2-TI2V-5B \
    --lr_ckpt <YOUR_PATH>/lr_anchor_checkpoint/model.pt \
    --hr_ckpt <YOUR_PATH>/hr_2k_checkpoint/model.pt \
    --prompt_file <YOUR_PATH>/prompts.txt \
    --video_dir <YOUR_PATH>/outputs/videos_2k \
    --resolution 2k \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8
