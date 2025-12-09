# count_params_sr.py
import argparse
import os

import os

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")

import torch
from omegaconf import OmegaConf

from model_wan_trainer_upsample import WanModel_Trainer


def build_config():
    # 基本上照抄 train_sr.py，只是去掉和训练无关的东西
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_base_path", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Dataset_fps24")
    parser.add_argument("--dataset_metadata_path", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/UltraVideo/matched_short.json")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--data_file_keys", type=str, default=("clip_id",))
    parser.add_argument("--max_pixels", type=int, default=2560 * 1440)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--time_division_factor", type=int, default=4)
    parser.add_argument("--time_division_remainder", type=int, default=1)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--config_path", type=str, default="/root/ultrawan/configs/self_forcing_dmd.yaml")
    parser.add_argument("--logdir", type=str, default="/tmp/dummy_logdir")  # 随便给个路径
    parser.add_argument("--generator_ckpt", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_latent_up_2_time_modulation/checkpoint_model_003000/model.pt")
    parser.add_argument("--load_generator_ckpt", type=bool, default=True)
    parser.add_argument("--trainable_backbone", type=bool, default=True)
    parser.add_argument("--use_ema", type=bool, default=False)

    args, _ = parser.parse_known_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("/root/ultrawan/configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.dataset_base_path = args.dataset_base_path
    config.dataset_metadata_path = args.dataset_metadata_path
    config.dataset_repeat = args.dataset_repeat
    config.data_file_keys = args.data_file_keys
    config.max_pixels = args.max_pixels
    config.height = args.height
    config.width = args.width
    config.num_frames = args.num_frames
    config.time_division_factor = args.time_division_factor
    config.time_division_remainder = args.time_division_remainder
    config.use_gpu = args.use_gpu
    config.config_path = args.config_path
    config.generator_ckpt = args.generator_ckpt
    config.trainable_backbone = args.trainable_backbone
    config.load_generator_ckpt = args.load_generator_ckpt
    config.use_ema = args.use_ema

    time_part = int((config.num_frames - 1) // config.time_division_factor) + 1
    config.seq_len = int(config.height // 32) * int(config.width // 32) * time_part
    config.sr_mode = True

    return config


def count_module_params(module: torch.nn.Module, name: str):
    num = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"{name:25s}: {num/1e6:8.2f} M ({num/1e9:.4f} B)")
    return num


def main():
    config = build_config()
    trainer = WanModel_Trainer(config)
    model = trainer.model  # SelfForcingWan_Upsample

    total = 0
    print("===== Trainable parameters by part =====")
    total += count_module_params(model.generator, "generator")
    total += count_module_params(model.text_encoder, "text_encoder")
    total += count_module_params(model.vae, "vae")

    # 如果想看 generator 内部，也可以再拆：
    if hasattr(model.generator, "model"):
        total += 0  # 不重复加，只做展示
        print("\n--- generator.model breakdown ---")
        for name, sub in model.generator.model.named_children():
            n = sum(p.numel() for p in sub.parameters() if p.requires_grad)
            print(f"generator.model.{name:15s}: {n/1e6:8.2f} M ({n/1e9:.4f} B)")

    print("\n===== TOTAL trainable params =====")
    print(f"Total trainable        : {total:,} (~{total/1e9:.4f} B)")


if __name__ == "__main__":
    main()
