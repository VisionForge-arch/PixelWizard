import torch
import torch.multiprocessing as mp
import time
import os

def occupy_single_gpu(rank, size, sleep):
    torch.cuda.set_device(rank)
    print(f"[GPU {rank}] Starting worker...")

    # Allocate large tensors
    a = torch.randn((size, size), device=rank)
    b = torch.randn((size, size), device=rank)

    print(f"[GPU {rank}] Allocated {size}x{size} tensors")

    # Infinite compute loop to keep GPU busy
    while True:
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        if sleep > 0:
            time.sleep(sleep)


def main(size=8192, sleep=0.0):
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")

    # Spawn one process per GPU
    mp.spawn(
        occupy_single_gpu,
        args=(size, sleep),
        nprocs=num_gpus,
        join=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--sleep", type=float, default=0.0)

    args = parser.parse_args()

    main(size=args.size, sleep=args.sleep)