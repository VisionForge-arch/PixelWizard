torchrun --nproc_per_node=8 \
    generate_multiple2.py \
    --size=448*256 \
    --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/eval_100/240p_5s/pt \
    --wan_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_240p/checkpoint_model_000100/model.pt \
    --use_ema \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \