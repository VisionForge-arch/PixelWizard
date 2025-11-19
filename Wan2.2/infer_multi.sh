torchrun --nproc_per_node=8 \
    generate_multiple_i2v.py \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \