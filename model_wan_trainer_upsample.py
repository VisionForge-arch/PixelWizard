import gc
import logging
from PIL import Image
import torchvision.transforms as transforms

from model_wan import SelfForcingWan_Upsample
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
import torch.nn.functional as F
import torchvision.transforms.functional as TF

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
        self.model = SelfForcingWan_Upsample(config, device=self.device)
        
        raw_trainable = sum(p.numel() for p in self.model.generator.parameters() if p.requires_grad)
        if self.is_main_process:
            print(f"[pre-FSDP] generator trainable params: {raw_trainable/1e6:.2f}M "
                  f"(~{raw_trainable/1e9:.4f}B)")
        
        
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

        
        # =========== OPTIMIZER =====
        shard_trainable = sum(p.numel() for p in self.model.generator.parameters() if p.requires_grad)
        print(f"[post-FSDP] local shard params: {shard_trainable/1e6:.2f}M")
        
        
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        
        # ============ EMA =============
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
            

        self.generator_ema = None
        if config.use_ema and (config.ema_weight > 0.0):
            print(f"Setting up EMA with weight {config.ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=config.ema_weight)
            
            
        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "load_generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(state_dict, strict=True)
            

        ##############################################################################################################

        
        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        # ============ SAVE ============
        self.output_path = config.logdir
        if self.is_main_process:
            wandb.init(
                project="self_forcing_long",
                name="self_forcing_long", 
                config=dict(config)
            )
        self.previous_time = None
        


    def random_degrade(self, hr_frames, target_size=(480, 832)):
        
        
        # ========== 瓶颈缩放 =============
        down_factor = random.uniform(8, 32)  # 2K (2048) / 32 = 64 pixel，非常糊
        
        # 计算瓶颈尺寸
        h, w = hr_frames.shape[-2:]
        bottleneck_h = int(h / down_factor)
        bottleneck_w = int(w / down_factor)
        
        # 1. 缩下去 (使用 area 或 bilinear 保证平滑，不要由 bicubic 产生伪影)
        tiny_frames = F.interpolate(hr_frames, size=(bottleneck_h, bottleneck_w), mode='area')
        
        # 2. 拉回 480p (使用 bilinear 保持模糊感，bicubic 会尝试锐化，不好)
        guidance = F.interpolate(tiny_frames, size=target_size, mode='bilinear', align_corners=False)
        
        # ======= 高斯模糊 =============
        k = random.choice([7, 9, 11])
        sigma = random.uniform(3.0, 5.0)
        guidance = TF.gaussian_blur(guidance, kernel_size=k, sigma=sigma)
        
        
        # ======= 噪声破坏 =============
        aug_level = 0.0
        if random.random() < 0.5:
            # 这里的噪声是为了破坏“像素级对应关系”，强迫模型关注语义
            aug_level = random.uniform(0.0, 0.1) # 0.1 已经很大了
            noise = torch.randn_like(guidance) * aug_level
            guidance = guidance + noise
    
        return guidance
    
    

    def train_one_step(self, batch):
        self.model.train()
        
        if self.step % 20 == 0:
            torch.cuda.empty_cache()
            
        # Step 1: Get the next batch of text prompts
        options = ['detailed_description', 'brief_description', 'summarized_description']
        text_prompts = batch[random.choice(options)]
        
        #text_prompts = batch["detailed_description"]

        # 转换PIL图像列表为tensor格式
        frames = batch["clip_id"].to(device=self.device, dtype=self.dtype)
        
        
        # 处理frames为480p，作为lr guidance。
        B, C, T, H, W = frames.shape
        
        frames_lr = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)   # [B*T, C, H, W]
     
        degrade_size = (240, 416)
        h_d, w_d = degrade_size
        frames_480p = self.random_degrade(frames_lr, target_size=degrade_size)  # 低质引导再退化
        
        
        
        frames_480p = frames_480p.reshape(B, T, C, h_d, w_d).permute(0, 2, 1, 3, 4)   # [b, C, T, h, w]
        
        
        # vae编码
        with torch.no_grad():
            
            clean_latent = self.model.vae.encode_to_latent(frames).to(device=self.device, dtype=self.dtype)           # [B, T, C, 90, 160]
            clean_latent_lr = self.model.vae.encode_to_latent(frames_480p).to(device=self.device, dtype=self.dtype)   # [B, T, C, 30, 52]
            
            
            B, T, C, H, W   = clean_latent.shape
            _, _, _, h, w   = clean_latent_lr.shape

            clean_latent_lr = clean_latent_lr.reshape(B*T, C, h, w)  # [B*T, C, h, w]
            clean_latent_lr = F.interpolate(clean_latent_lr, size=(H, W), mode='bilinear', align_corners=False)  # 可加 antialias=True（若版本支持）
            clean_latent_lr = clean_latent_lr.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
            
                    
            # print(f"frames.shape: {frames.shape}, clean_latent.shape: {clean_latent.shape}")
            # print(f"frames_480p.shape: {frames_480p.shape}, clean_latent_lr.shape: {clean_latent_lr.shape}")
                
        # VAE编码完成后立即释放frames显存
        del frames
        del frames_480p
        torch.cuda.empty_cache()
        

        batch_size = len(text_prompts)
        image_or_video_shape = clean_latent.shape

        # Extract the conditional infos
        with torch.no_grad():
            # 'promot_embeds': [B, 512, 4096]
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            
            
            uncond_proba = getattr(self.config, "uncond_proba", 0.1)
            if uncond_proba > 0:
                if not getattr(self, "unconditional_dict", None):
                    unconditional_dict = self.model.text_encoder(
                        text_prompts=[""] * batch_size)
                    unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                    self.unconditional_dict = unconditional_dict  # cache
                else:
                    unconditional_dict = self.unconditional_dict
            else:
                unconditional_dict = None
                
                
        generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                clean_latent_lr=clean_latent_lr
            )
        
        
        generator_loss.backward()
        
        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
        
        
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
            
            if self.generator_ema is not None:
                self.generator_ema.update(self.model.generator)
            
        
            self.step += 1
            
            
            # =========== Create EMA params ==============
            if self.config.use_ema and (self.step >= self.config.ema_start_step) and (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
        
            # =========== Save the model ================
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

        if self.config.use_ema and (self.config.ema_start_step < self.step):
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }    
        else:
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