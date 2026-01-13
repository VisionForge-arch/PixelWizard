export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
torchrun --standalone --nproc_per_node=6 \
    generate_multiple_upsample_shortcut.py \
    --size=2560*1440 \
    --sample_steps=4 \
    --prompt_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/eval_100_match_240.json \
    --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/2k_shortcut_new2/checkpoint_model_000350/model.pt \
    --save_file=/mnt/nas01-ak/IndividualDirs/wenxueli/eval_100/2k_shortcut_100/pt \
    --frame_num=121 \
    --sample_shift=5.5 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 6 \


# torchrun --standalone --nproc_per_node=6 \
#     generate_multiple_upsample_shortcut.py \
#     --size=2560*1440 \
#     --sample_steps=4 \
#     --prompt_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/woman_240p_pt_20.json \
#     --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/2k_shortcut_new2/checkpoint_model_000050/model.pt \
#     --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/240p_upsample_2k_shortcut2/pt_woman_20 \
#     --frame_num=121 \
#     --sample_shift=5.5 \
#     --dit_fsdp \
#     --t5_fsdp \
#     --ulysses_size 6 \


# torchrun --standalone --nproc_per_node=6 \
#     generate_multiple_upsample_shortcut.py \
#     --size=2560*1440 \
#     --sample_steps=4 \
#     --prompt_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/env_240p_pt.json \
#     --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/2k_shortcut_new2/checkpoint_model_000050/model.pt \
#     --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/240p_upsample_2k_shortcut2/pt_env_20 \
#     --frame_num=121 \
#     --sample_shift=5.5 \
#     --dit_fsdp \
#     --t5_fsdp \
#     --ulysses_size 6 \


# torchrun --standalone --nproc_per_node=8 \
#     generate_multiple_upsample_shortcut.py \
#     --size=2560*1440 \
#     --sample_steps=4 \
#     --prompt_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/prompt_to_file_240p.json \
#     --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/2k_shortcut/checkpoint_model_000800/model.pt \
#     --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/240p_upsample_2k_shortcut/pt3 \
#     --frame_num=121 \
#     --sample_shift=5.5 \
#     --use_ema \
#     --dit_fsdp \
#     --t5_fsdp \
#     --ulysses_size 8 \