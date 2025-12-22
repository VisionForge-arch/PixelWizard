torchrun --nproc_per_node=8 \
    generate_multiple2.py \
    --size=2560*1440 \
    --save_fine=/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/eval_100/2k_5s/pt \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 8 \