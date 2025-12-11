torchrun --standalone --nproc_per_node=4 \
    generate_multiple_upsample.py \
    --size=3840*2144 \
    --sample_steps=15 \
    --prompt_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompt_to_file_720p.json \
    --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_latent_up_2_time_modulation_4k/checkpoint_model_000100/model.pt \
    --frame_num=121 \
    --use_ema \
    --sample_shift=5.5 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 4 \