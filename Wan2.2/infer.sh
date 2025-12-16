torchrun --nproc_per_node=4 \
    generate_multiple.py \
    --wan_ckpt /mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_240p/checkpoint_model_005700/model.pt \
    --use_ema \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 4 \