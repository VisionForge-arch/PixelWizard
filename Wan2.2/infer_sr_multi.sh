torchrun --nproc_per_node=4 \
    generate_multiple_sr.py \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 4 \