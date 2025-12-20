torchrun --standalone --nproc_per_node=4 \
    generate_multiple_upsample_shortcut.py \
    --size=2560*1440 \
    --sample_steps=5 \
    --prompt_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompt_to_file_240p.json \
    --frame_num=121 \
    --use_ema \
    --sample_shift=5.5 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 4 \