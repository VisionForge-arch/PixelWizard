# export WANDB_MODE=offline
# # 在所有节点都设置这些变量
# export MASTER_ADDR=10.44.140.186  # 主节点 IP
# export MASTER_PORT=29500            # 通讯端口
# export NNODES=2                          # 总节点数


# torchrun \
#   --nnodes=${NNODES} \
#   --node_rank=${NODE_RANK} \
#   --nproc_per_node=8 \
#   --master_addr=${MASTER_ADDR} \
#   --master_port=${MASTER_PORT} \
#   train_lr.py \
#   --use_ema   


export WANDB_MODE=offline


torchrun \
  --nproc_per_node=8 \
  train_lr.py \
  --use_ema   

