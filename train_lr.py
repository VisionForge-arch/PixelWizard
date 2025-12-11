import argparse
import os
from tqdm import tqdm


from omegaconf import OmegaConf

import wandb
from model_wan_trainer_lr import WanModel_Trainer


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_base_path", type=str, default=None)
    parser.add_argument("--dataset_metadata_path", type=str, default="/mnt/vision-gen-ks3/Video_Generation/DataSets/vidgen-1M-sub/jsons/VidGen_1M_video3_recaption.jsonl")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--data_file_keys", type=str, default="video_path")
    parser.add_argument("--max_pixels", type=int, default=640*352)    #  2560*1440, 
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--time_division_factor", type=int, default=4)
    parser.add_argument("--time_division_remainder", type=int, default=1)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--config_path", type=str, default="/root/ultrawan/configs/self_forcing_dmd.yaml")
    parser.add_argument("--logdir", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/Ultra_Train_Weight/wan_360p")
    parser.add_argument("--generator_ckpt", type=str, default=None)
    parser.add_argument("--load_generator_ckpt", action="store_true")
    parser.add_argument("--trainable_backbone", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    
    args = parser.parse_args()
    
    
    
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("/root/ultrawan/configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    # get the filename of config_path
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
    config.logdir = args.logdir
    config.generator_ckpt = args.generator_ckpt
    config.trainable_backbone = args.trainable_backbone
    config.load_generator_ckpt = args.load_generator_ckpt
    config.use_ema = args.use_ema
    
    time_part = int((config.num_frames - 1) // config.time_division_factor) + 1
    config.seq_len = int(config.height // 32) * int(config.width // 32) * time_part
    config.sr_mode = False
    
    # =========init dataset=========
    
    model_trainer = WanModel_Trainer(config)
    model_trainer.train()
    
    wandb.finish()
    
    
if __name__ == "__main__":
    main()
