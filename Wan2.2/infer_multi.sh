
torchrun --nproc_per_node=8 \
    generate_multiple.py \
    --use_ema \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \