torchrun --nproc_per_node=8 \
    generate_multiple_upsample.py \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \