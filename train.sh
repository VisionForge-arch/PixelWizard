export CUDA_VISIBLE_DEVICES=1,2,3
export WANDB_MODE=offline

torchrun --nproc_per_node=3 \
  train.py \