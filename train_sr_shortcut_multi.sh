export WANDB_MODE=disabled
# 在所有节点都设置这些变量
export MASTER_ADDR=10.44.142.56  # 主节点 IP
export MASTER_PORT=29500            # 通讯端口
export NNODES=2                          # 总节点数

torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=8 \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train_sr.py \
  --height=2144 \
  --width=3840 \
  --num_frames=81 \
  --config_path=/root/ultrawan/configs/self_forcing_dmd_4k.yaml \
  --logdir=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/4k_shortcut_new2 \
  --generator_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/4k_shortcut_new/checkpoint_model_000500/model.pt \
  --trainable_backbone \
  --shortcut \
  # --load_generator_ckpt \


# torchrun --standalone --nproc_per_node=8 \
#   train_sr.py \
#   --height=1440 \
#   --width=2560 \
#   --num_frames=69 \
#   --config_path=/root/ultrawan/configs/self_forcing_dmd.yaml \
#   --logdir=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/2k_shortcut_new \
#   --generator_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_latent_up_2_time_modulation2/checkpoint_model_000800/model.pt \
#   --trainable_backbone \
#   --shortcut \