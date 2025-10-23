import argparse
import os
from tqdm import tqdm


from omegaconf import OmegaConf

import wandb
from model_wan_trainer import WanModel_Trainer
    
    


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--data_file_keys", type=str, default=("clip_id",))
    # parser.add_argument("--max_pixels", type=int, default=2560*1440)
    # parser.add_argument("--height", type=int, default=1440)
    # parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--max_pixels", type=int, default=224*224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--time_division_factor", type=int, default=4)
    parser.add_argument("--time_division_remainder", type=int, default=1)
    parser.add_argument("--use_gpu", type=bool, default=False)
    parser.add_argument("--config_path", type=str, default="/hpc2hdd/home/htian395/Wenxue/Underwater/Underwater_Video_UIE/configs/self_forcing_dmd.yaml")
    parser.add_argument("--logdir", type=str, default="/hpc2hdd/home/htian395/Wenxue/Underwater/Underwater_Video_UIE/logs/self_forcing_dmd")
    parser.add_argument('--input_root', type=str, default='/hpc2ssd/JH_DATA/spooler/htian395/C-underwater/0-dataset-split/UIED-no-rename/train/raw-890')
    parser.add_argument('--gt_root', type=str, default='/hpc2ssd/JH_DATA/spooler/htian395/C-underwater/0-dataset-split/UIED-no-rename/train/reference-890')
    args = parser.parse_args()
    
    
    
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("/hpc2hdd/home/htian395/Wenxue/Underwater/Underwater_Video_UIE/configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.input_root = args.input_root
    config.gt_root = args.gt_root
    config.max_pixels = args.max_pixels
    config.height = args.height
    config.width = args.width
    config.num_frames = args.num_frames
    config.time_division_factor = args.time_division_factor
    config.time_division_remainder = args.time_division_remainder
    config.use_gpu = args.use_gpu
    config.config_path = args.config_path
    config.logdir = args.logdir
    
    config.seq_len = int(config.height // 16) * int(config.width // 16)
    
    # =========init dataset=========
    
    
    model_trainer = WanModel_Trainer(config)
    model_trainer.train()
    
    wandb.finish()
    
    
if __name__ == "__main__":
    main()
