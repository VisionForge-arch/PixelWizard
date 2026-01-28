
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


torchrun --standalone --nproc_per_node=4 \
    generate_multiple2.py \
    --size=448*256 \
    --save_file=/mnt/nas01-ak/IndividualDirs/wenxueli/eval_100/eval_100/240p_5s_no_train/pt \
    --base_seed=10 \
    --sample_shift=4.3 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 4 \
