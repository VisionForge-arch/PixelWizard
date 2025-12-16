import torch
import torch.multiprocessing as mp
import time
import os


def occupy_single_gpu(rank, size, sleep, gpu_ids=None):
    """
    rank: 进程内的序号（0 ~ nprocs-1）
    gpu_ids: 实际要使用的 GPU ID 列表，比如 [0, 2]，则 rank=0 用 0 号卡，rank=1 用 2 号卡
    """
    if gpu_ids is None:
        device_id = rank
    else:
        device_id = gpu_ids[rank]

    torch.cuda.set_device(device_id)
    print(f"[GPU {device_id}] Starting worker...")

    # Allocate large tensors
    a = torch.randn((size, size), device=device_id)
    b = torch.randn((size, size), device=device_id)

    print(f"[GPU {device_id}] Allocated {size}x{size} tensors")

    # Infinite compute loop to keep GPU busy
    while True:
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        if sleep > 0:
            time.sleep(sleep)


def main(size=8192, sleep=0.0, gpus=None):
    """
    gpus: 字符串形式的 GPU ID 列表，比如 "0,1,3"；如果为 None，则使用所有可见 GPU
    """
    if gpus is not None:
        gpu_ids = [int(x) for x in gpus.split(",") if x.strip() != ""]
        num_gpus = len(gpu_ids)
    else:
        num_gpus = torch.cuda.device_count()
        gpu_ids = None

    print(f"Detected {torch.cuda.device_count()} GPUs, using {num_gpus} GPUs")

    # Spawn one process per GPU
    mp.spawn(
        occupy_single_gpu,
        args=(size, sleep, gpu_ids),
        nprocs=num_gpus,
        join=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="要占用的 GPU ID，例如: 0 或 0,1,3；不填则使用所有可见 GPU",
    )

    args = parser.parse_args()

    main(size=args.size, sleep=args.sleep, gpus=args.gpus)