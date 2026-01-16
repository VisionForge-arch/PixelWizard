export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# torchrun --nproc_per_node=8 \
#     generate_multiple2.py \
#     --size=448*256 \
#     --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/eval_100/240p_5s/pt_new \
#     --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_240p/checkpoint_model_000100/model.pt \
#     --use_ema \
#     --sample_shift=5.0 \
#     --dit_fsdp \
#     --t5_fsdp \
#     --ulysses_size 8 \


torchrun --nproc_per_node=6 \
    generate_multiple2.py \
    --size=448*256 \
    --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/eval_100/240p_5s/pt_new \
    --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_240p_new/checkpoint_model_001800/model.pt \
    --sample_shift=4.5 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 6 \
