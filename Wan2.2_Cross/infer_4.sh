# torchrun --standalone --nproc_per_node=8 \
#     generate_multiple_upsample_shortcut.py \
#     --size=2560*1440 \
#     --sample_steps=5 \
#     --prompt_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/prompt_to_file_240p.json \
#     --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/2k_shortcut2/checkpoint_model_000400/model.pt \
#     --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/240p_upsample_2k_shortcut2/pt \
#     --frame_num=121 \
#     --sample_shift=5.5 \
#     --dit_fsdp \
#     --t5_fsdp \
#     --ulysses_size 8 \


torchrun --standalone --nproc_per_node=8 \
    generate_multiple_upsample_shortcut.py \
    --size=2560*1440 \
    --sample_steps=4 \
    --prompt_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/prompt_to_file_240p.json \
    --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/2k_shortcut/checkpoint_model_000800/model.pt \
    --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/240p_upsample_2k_shortcut/pt3 \
    --frame_num=121 \
    --sample_shift=1 \
    --sample_guide_scale=3 \
    --use_ema \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \