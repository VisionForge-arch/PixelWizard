export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline

# torchrun --standalone --nproc_per_node=8 \
#   train_sr.py \
#   --height=2144 \
#   --width=3840 \
#   --num_frames=69 \
#   --config_path=/root/ultrawan/configs/self_forcing_dmd_4k.yaml \
#   --logdir=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/4k_shortcut_new \
#   --generator_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/2k_shortcut_new/checkpoint_model_001900/model.pt \
#   --trainable_backbone \
#   --shortcut \
#   --load_generator_ckpt \


torchrun --standalone --nproc_per_node=8 \
  train_sr.py \
  --height=1440 \
  --width=2560 \
  --num_frames=121 \
  --config_path=/root/ultrawan/configs/self_forcing_dmd.yaml \
  --logdir=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/2k_shortcut_new2 \
  --generator_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/2k_shortcut_new2/checkpoint_model_000050/model.pt \
  --trainable_backbone \
  --shortcut \
  --load_generator_ckpt \