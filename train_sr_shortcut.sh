export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline

torchrun --standalone --nproc_per_node=8 \
  train_sr.py \
  --max_pixels=3840*2144 \
  --height=2144 \
  --width=3840 \
  --config_path=/root/ultrawan/configs/self_forcing_dmd_4k.yaml \
  --logdir=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/4k_shortcut \
  --generator_ckpt=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_latent_up_2_time_modulation2/checkpoint_model_000800/model.pt \
  --trainable_backbone \
  --shortcut \
  --load_generator_ckpt \