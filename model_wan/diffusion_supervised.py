from typing import Tuple
import torch
import torch.nn.functional as F

from model_wan.base import BaseModel, SelfForcingModel
from utils_long.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from pipeline_long import SelfForcingTrainingPipeline

class SupervisedSelfForcingWan(SelfForcingModel):
    def __init__(self, args, device):
        """
        支持监督训练的Self-Forcing模型
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 39)
        
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
            
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Step 2: Initialize all hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        
        # 4步timestep
        self.training_timesteps = [250, 500, 750, 1000]
        
        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None
            
    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)
        
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def simple_flow_matching_loss(self, conditional_dict: dict, clean_latent: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        简单的Flow Matching训练，不使用复杂的自回归pipeline
        适合短视频或单独的视频片段训练
        """
        batch_size, num_frame = clean_latent.shape[:2]
        
        # 从4个timestep中随机选择
        timestep_idx = torch.randint(0, len(self.training_timesteps), (batch_size, num_frame), device=self.device)
        timestep = torch.tensor([self.training_timesteps[i] for i in timestep_idx.flatten()], 
                               device=self.device, dtype=torch.long).reshape(batch_size, num_frame)
        
        # Flow Matching训练
        noise = torch.randn_like(clean_latent)
        noisy_latent = self.scheduler.add_noise(
            clean_latent.flatten(0, 1), 
            noise.flatten(0, 1), 
            timestep.flatten(0, 1)
        ).unflatten(0, clean_latent.shape[:2])
        
        # 直接预测，不用KV cache（适合短视频）
        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latent,
            conditional_dict=conditional_dict,
            timestep=timestep
        )
        
        # 计算loss
        training_target = self.scheduler.training_target(clean_latent, noise, timestep)
        loss = F.mse_loss(flow_pred, training_target)
        
        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach(),
            "timestep": timestep.float().mean().detach()
        }
        return loss, log_dict

    def autoregressive_supervised_loss(self, image_or_video_shape, conditional_dict: dict, clean_latent: torch.Tensor, initial_latent: torch.Tensor = None) -> Tuple[torch.Tensor, dict]:
        """
        自回归监督训练：保持block-by-block结构，但用真实数据监督
        适合长视频训练
        """
        batch_size, num_frame = image_or_video_shape[:2]
        
        # 选择一个统一的timestep（方案A）
        timestep_value = torch.randint(0, len(self.training_timesteps), (1,)).item()
        timestep_value = self.training_timesteps[timestep_value]
        
        total_loss = 0
        num_blocks = 0
        
        # 按block处理
        block_size = self.num_frame_per_block
        start_idx = 1 if initial_latent is not None else 0  # 如果有initial frame就跳过第一帧
        
        for i in range(start_idx, num_frame, block_size):
            end_idx = min(i + block_size, num_frame)
            current_block_size = end_idx - i
            
            if current_block_size <= 0:
                continue
                
            # 当前block的真实数据
            current_clean = clean_latent[:, i:end_idx]
            current_conditional = {k: v for k, v in conditional_dict.items()}
            
            # 添加initial_latent到条件中（模拟KV cache的作用）
            if i > start_idx:
                # 使用前面生成的结果作为context
                context_frames = clean_latent[:, max(0, i-4):i]  # 取前面4帧作为context
                current_conditional["context_frames"] = context_frames
            elif initial_latent is not None:
                current_conditional["initial_latent"] = initial_latent
            
            # Flow Matching训练
            timestep = torch.ones((batch_size, current_block_size), device=self.device, dtype=torch.long) * timestep_value
            
            noise = torch.randn_like(current_clean)
            noisy_latent = self.scheduler.add_noise(
                current_clean.flatten(0, 1), 
                noise.flatten(0, 1), 
                timestep.flatten(0, 1)
            ).unflatten(0, current_clean.shape[:2])
            
            # 预测
            flow_pred, x0_pred = self.generator(
                noisy_image_or_video=noisy_latent,
                conditional_dict=current_conditional,
                timestep=timestep
            )
            
            # 计算损失
            training_target = self.scheduler.training_target(current_clean, noise, timestep)
            block_loss = F.mse_loss(flow_pred, training_target)
            total_loss += block_loss
            num_blocks += 1
        
        avg_loss = total_loss / max(num_blocks, 1)
        
        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": clean_latent.detach(),  # 简化log
            "num_blocks": num_blocks,
            "timestep": timestep_value
        }
        return avg_loss, log_dict

    def generator_loss(self, image_or_video_shape, conditional_dict: dict, unconditional_dict: dict, clean_latent: torch.Tensor, initial_latent: torch.Tensor = None) -> Tuple[torch.Tensor, dict]:
        """
        混合训练策略：根据视频长度选择训练方式
        """
        batch_size, num_frame = image_or_video_shape[:2]
        
        # 策略选择
        if num_frame <= 16:  # 短视频用简单Flow Matching
            return self.simple_flow_matching_loss(conditional_dict, clean_latent)
        else:  # 长视频用自回归监督训练
            return self.autoregressive_supervised_loss(image_or_video_shape, conditional_dict, clean_latent, initial_latent)
