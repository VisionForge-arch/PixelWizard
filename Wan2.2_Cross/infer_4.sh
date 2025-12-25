torchrun --standalone --nproc_per_node=8 \
    generate_multiple_upsample_shortcut.py \
    --size=2560*1440 \
    --sample_steps=4 \
    --prompt_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/prompt_to_file_240p.json \
    --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/2k_shortcut/checkpoint_model_001000/model.pt \
    --frame_num=121 \
    --use_ema \
    --sample_shift=3 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \