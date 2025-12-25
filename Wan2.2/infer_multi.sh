
torchrun --nproc_per_node=8 \
    generate_multiple.py \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \