# torchrun --nproc_per_node=8 \
#     generate_multiple.py \
#     --use_ema \
#     --dit_fsdp \
#     --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/360p_woman/pt \
#     --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_360p/checkpoint_model_004200/model.pt \
#     --size=640*352 \
#     --t5_fsdp \
#     --ulysses_size 8 \


# torchrun --nproc_per_node=8 \
#     generate_multiple.py \
#     --sample_steps=20 \
#     --base_seed=1 \
#     --use_ema \
#     --dit_fsdp \
#     --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/240p_woman/pt_10 \
#     --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_240p/checkpoint_model_000100/model.pt \
#     --size=448*256 \
#     --t5_fsdp \
#     --ulysses_size 8 \

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

torchrun --standalone --nproc_per_node=6 \
    generate_multiple.py \
    --sample_steps=50 \
    --base_seed=42 \
    --use_ema \
    --dit_fsdp \
    --prompt_file=/root/ultrawan/Wan2.2/prompt4.txt \
    --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/240p_niren/pt_1 \
    --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_240p/checkpoint_model_000100/model.pt \
    --size=448*256 \
    --t5_fsdp \
    --offload_model=False \
    --ulysses_size 6 \