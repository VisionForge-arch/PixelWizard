export WANDB_MODE=offline

torchrun --standalone --nproc_per_node=8 \
  train_lr.py \
  --use_ema   \