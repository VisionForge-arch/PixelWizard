torchrun --standalone --nproc_per_node=8 \
    generate_multiple_upsample.py \
    --size=2560*1440 \
    --sample_steps=2 \
    --prompt_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/prompt_to_file_240p.json \
    --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_latent_up_2_time_modulation2/checkpoint_model_000800/model.pt \
    --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/240p_upsample_2k/pt_woman \
    --frame_num=121 \
    --use_ema \
    --sample_shift=5.5 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \



# torchrun --standalone --nproc_per_node=8 \
#     generate_multiple_upsample.py \
#     --size=3840*2144 \
#     --sample_steps=15 \
#     --prompt_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompt_to_file_360p.json \
#     --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_latent_up_2_time_modulation_4k/checkpoint_model_000600/model.pt \
#     --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/360p_upsample_4k \
#     --frame_num=121 \
#     --use_ema \
#     --sample_shift=5.5 \
#     --dit_fsdp \
#     --t5_fsdp \
#     --ulysses_size 8 \