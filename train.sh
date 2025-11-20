export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline

torchrun --nproc_per_node=8 \
  train_sr.py \