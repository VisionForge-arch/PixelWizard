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
import random
import wandb
from utils_long.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
import torch.distributed as dist
from dataset import UnifiedDataset, cycle

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
        
        dataset = UnifiedDataset(
            base_path=config.dataset_base_path,
            metadata_path=config.dataset_metadata_path,
            repeat=config.dataset_repeat,
            data_file_keys=config.data_file_keys,
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=config.dataset_base_path,
                max_pixels=config.max_pixels,
                height=config.height,
                width=config.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=config.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
                ),
            )
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
        
    '''
    def convert_pil_list_to_tensor(self, pil_frames):
        """
        将PIL图像列表转换为VAE期望的tensor格式
        Args:
            pil_frames: List[PIL.Image] - PIL图像列表
        Returns:
            torch.Tensor: shape [1, 3, num_frames, height, width]
        """
        # 转换PIL图像为tensor
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为 [0,1] 范围的tensor，形状 [C, H, W]
        ])
        
        # 转换每一帧
        tensor_frames = []
        for pil_frame in pil_frames:
            frame_tensor = transform(pil_frame)  # [3, H, W]
            tensor_frames.append(frame_tensor)
        
        # 堆叠所有帧：[num_frames, 3, H, W]
        video_tensor = torch.stack(tensor_frames, dim=0)
        
        # 重新排列为VAE期望的格式：[1, 3, num_frames, H, W]
        # 从 [num_frames, 3, H, W] 到 [1, 3, num_frames, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        
        return video_tensor
    '''



    def train_one_step(self, batch):
        self.model.train()
        
        if self.step % 20 == 0:
            torch.cuda.empty_cache()
            
        # Step 1: Get the next batch of text prompts
        # DataLoader会将字符串打包成列表，直接使用
        options = ['detailed_description', 'brief_description', 'summarized_description']
        text_prompts = batch[random.choice(options)]
        
        #text_prompts = batch["detailed_description"]

        # 转换PIL图像列表为tensor格式
        frames = batch["clip_id"].to(device=self.device, dtype=self.dtype)
        
        with torch.no_grad():
            clean_latent = self.model.vae.encode_to_latent(frames).to(device=self.device, dtype=self.dtype)   
            
            if self.step % 100 == 0:  # 每100步打印一次
                print(f"frames.shape: {frames.shape}, clean_latent.shape: {clean_latent.shape}")
        
        # VAE编码完成后立即释放frames显存
        del frames
        torch.cuda.empty_cache()

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            # 'promot_embeds': [B, 512, 4096]
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict
                
        generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
            )
        
        print("generator_loss:", generator_loss.mean().item())
        
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