torchrun --nproc_per_node=4 \
    generate.py \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 4 \