export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

torchrun --nproc_per_node=7 \
    generate_multiple.py \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 7 \