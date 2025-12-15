export WANDB_MODE=offline
# 在所有节点都设置这些变量
export MASTER_ADDR=10.44.140.186  # 主节点 IP
export MASTER_PORT=29500            # 通讯端口
export NNODES=2                          # 总节点数
export NODE_RANK=0                      # 当前节点序号，主节点0，从节点1,2,...


torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=8 \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train_lr.py \
  --use_ema   
