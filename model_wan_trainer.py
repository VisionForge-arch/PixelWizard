import gc
import logging
from PIL import Image
import torchvision.transforms as transforms

from model_wan import SelfForcingWan
import torch
from utils_long.misc import (
    set_seed,
    merge_dict_list
)
import os
import time
import wandb
from utils_long.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
import torch.distributed as dist
from dataset import cycle
from dataset_image import UIEBD_Dataset_Wrapper

class WanModel_Trainer:
    def __init__(self, config):
        
        self.step = 0
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.is_main_process = global_rank == 0
        self.config = config
        self.device = "cuda"
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        
        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)
        
        dataset = UIEBD_Dataset_Wrapper(config.input_root, config.gt_root, 'train')
        print("len(dataset):", len(dataset))
        
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)
        
        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        # Step 2: Initialize the model and optimizer
        self.model = SelfForcingWan(config, device=self.device)
        
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )


        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy
        )
        self.model.vae = self.model.vae.to(device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)




        # 设置训练参数
        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 1.0)

        
        # ===== optimizer =====
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # ========= save ==========
        self.output_path = config.logdir
        if self.is_main_process:
            wandb.init(
                project="self_forcing_long",
                name="self_forcing_long", 
                config=dict(config)
            )
        self.previous_time = None
        

    def train_one_step(self, batch):
        self.model.train()
        
        if self.step % 20 == 0:
            torch.cuda.empty_cache()
            
        input_image = batch[0]
        target_image = batch[1]
            
        # Step 1: Get the next batch of text prompts
        # DataLoader会将字符串打包成列表，直接使用
        #text_prompts = batch["detailed_description"]
        text_prompts = [""]

        # 转换PIL图像列表为tensor格式
        frames = target_image.to(device=self.device, dtype=self.dtype)
        input_frames = input_image.to(device=self.device, dtype=self.dtype)
        frames = frames.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
        input_frames = input_frames.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
        
        with torch.no_grad():
            clean_latent = self.model.vae.encode_to_latent(frames)   # [1, 1, 48, 15, 15]
            input_latent = self.model.vae.encode_to_latent(input_frames)   # [1, 1, 48, 15, 15]
            # 确保张量正确分离并转换类型
            clean_latent = clean_latent.detach().to(dtype=self.dtype)
            input_latent = input_latent.detach().to(dtype=self.dtype)
            
            if self.step % 100 == 0:  # 每100步打印一次
                print(f"input_frames.shape: {input_latent.shape}, clean_latent.shape: {clean_latent.shape}")
        

        batch_size = target_image.shape[0]
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            # 'promot_embeds': [B, 512, 4096]
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
                
        generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                clean_latent=clean_latent,
                input_latent=input_latent,
            )
        print(f"generator_loss: {generator_loss}")
        generator_loss.backward()
        
        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
        
        # generator_grad_norm = torch.nn.utils.clip_grad_norm_(
        #     self.model.generator.parameters(), 
        #     self.max_grad_norm_generator
        # )  # 单卡用
        
        generator_log_dict.update({"generator_loss": generator_loss,
                                   "generator_grad_norm": generator_grad_norm})

        
        return generator_log_dict

    def train(self):
        start_step = self.step
        
        while True:
        
            extras_list = []
            self.generator_optimizer.zero_grad(set_to_none=True)
            batch = next(self.dataloader)
            
            extra = self.train_one_step(batch)
            #print(extra)
            extras_list.append(extra)
            generator_log_dict = merge_dict_list(extras_list)
            self.generator_optimizer.step()
        
            self.step += 1
        
            # Save the model
            if (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()
                
            if self.is_main_process:
                wandb_loss_dict = {}
                wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                        }
                    )
                wandb.log(wandb_loss_dict, step=self.step)
            
            # ======
            print("generator_loss of Step", self.step, ":", generator_log_dict["generator_loss"].mean().item())
    

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()
                
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
                
        
    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(self.model.generator)


        state_dict = {
            "generator": generator_state_dict,
        }
        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                    f"checkpoint_model_{self.step:06d}", "model.pt"))