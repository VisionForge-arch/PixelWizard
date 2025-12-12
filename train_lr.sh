export WANDB_MODE=offline

torchrun --standalone --nproc_per_node=8 \
  train_lr.py \
  --load_generator_ckpt \
  --use_ema 