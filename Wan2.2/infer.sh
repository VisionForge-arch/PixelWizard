torchrun --nproc_per_node=8 \
    generate_multiple2.py \
    --size=3840*2144 \
    --save_file=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/eval_100/4k_5s/pt \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \