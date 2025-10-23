export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline

torchrun --nproc_per_node=4 \
  train.py \