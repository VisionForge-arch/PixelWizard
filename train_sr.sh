export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export WANDB_MODE=offline

torchrun --standalone --nproc_per_node=7 \
  train_sr.py \
  --load_generator_ckpt \
  --trainable_backbone \
  --use_ema 