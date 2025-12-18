export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline

torchrun --standalone --nproc_per_node=4 \
  train_sr_causal.py \
  --load_generator_ckpt \
  --trainable_backbone \