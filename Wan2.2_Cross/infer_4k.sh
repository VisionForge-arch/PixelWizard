export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# torchrun --standalone --nproc_per_node=6 \
#     generate_multiple_upsample_shortcut.py \
#     --size=3840*2144 \
#     --sample_steps=4 \
#     --prompt_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/woman_240p_pt_20.json \
#     --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/4k_shortcut_new2/checkpoint_model_000050/model.pt \
#     --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/240p_upsample_4k_shortcut/pt_woman_20 \
#     --frame_num=121 \
#     --sample_shift=5.8 \
#     --dit_fsdp \
#     --t5_fsdp \
#     --ulysses_size 6 \

torchrun --standalone --nproc_per_node=6 \
    generate_multiple_upsample_shortcut.py \
    --eval_bench \
    --size=3840*2144 \
    --sample_steps=4 \
    --prompt_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/prompts.jsonl \
    --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/4k_shortcut_new2/checkpoint_model_000050/model.pt \
    --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/240p_upsample_4k_shortcut/pt_woman_20 \
    --frame_num=121 \
    --sample_shift=5.8 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 6 \