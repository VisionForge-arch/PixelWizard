from typing import Tuple
import torch
import torch.nn.functional as F

from model_wan.base import BaseModel, SelfForcingModel
from utils_long.wan2_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper2_2
from pipeline_long import SelfForcingTrainingPipeline


class SelfForcingWan(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the Diffusion loss module.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)
        
        # Random crop settings for loss computation (to save memory)
        self.use_random_crop = getattr(args, "use_random_crop", False)
        self.crop_height = getattr(args, "crop_height", 30)
        self.crop_width = getattr(args, "crop_width", 52)
        
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
        self.teacher_forcing = getattr(args, "teacher_forcing", False)
        
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)
        
        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None
            
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)

        ##############################################################################################################
        # (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        # if getattr(args, "generator_ckpt", False):
        #     print(f"Loading pretrained generator from {args.generator_ckpt}")
        #     state_dict = torch.load(args.generator_ckpt, map_location="cpu")
        #     if "generator" in state_dict:
        #         state_dict = state_dict["generator"]
        #     elif "model" in state_dict:
        #         state_dict = state_dict["model"]
        #     self.generator.load_state_dict(
        #         state_dict, strict=True
        #     )
            
        ##############################################################################################################

            
    def _initialize_models(self, args, device):
        self.seq_len = args.seq_len
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), seq_len=self.seq_len)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper2_2()
        self.vae.requires_grad_(False)
        
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def random_crop(self, *tensors):
        """
        随机裁剪多个tensor到指定大小，所有tensor使用相同的crop位置
        Input: tensors with shape [B, F, C, H, W]
        Output: cropped tensors with shape [B, F, C, crop_h, crop_w]
        """
        if not self.use_random_crop:
            return tensors
        
        # 获取原始尺寸
        _, _, _, h, w = tensors[0].shape
        
        # 确保crop尺寸不超过原始尺寸
        crop_h = min(self.crop_height, h)
        crop_w = min(self.crop_width, w)
        
        # 随机选择crop的起始位置
        top = torch.randint(0, h - crop_h + 1, (1,)).item() if h > crop_h else 0
        left = torch.randint(0, w - crop_w + 1, (1,)).item() if w > crop_w else 0
        
        # 对所有tensor应用相同的crop
        cropped = []
        for tensor in tensors:
            cropped.append(tensor[:, :, :, top:top+crop_h, left:left+crop_w])
        
        return tuple(cropped) if len(cropped) > 1 else cropped[0]

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        clean_latent_lr: torch.Tensor = None,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        noise = torch.randn_like(clean_latent)
        print(f"image_or_video_shape: {image_or_video_shape}")
        batch_size, num_frame = image_or_video_shape[:2]

        # Step 2: Randomly sample a timestep and add noise to denoiser inputs (Flow Matching)
        # 从[0, 1000)中随机采样timestep index
        index = torch.randint(
            0,
            len(self.scheduler.timesteps),
            (batch_size,),
            device=self.device,
            dtype=torch.long
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        timestep = timestep[:, None].expand(batch_size, num_frame)  # [B, F]
        # Flow Matching: x_t = (1-sigma) * x0 + sigma * noise
        
        if clean_latent_lr is not None:
            
            
            clean_latent_lr = F.interpolate(
                clean_latent_lr,
                size=clean_latent.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            print('===upsample===')
            print(f"clean_latent_lr.shape: {clean_latent_lr.shape}, clean_latent.shape: {clean_latent.shape}")
            
            noisy_latents = self.scheduler.add_noise(
                clean_latent_lr.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        else:
        
        
            noisy_latents = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        # Flow Matching target: v = noise - x0 (velocity field)
        training_target = self.scheduler.training_target(
            clean_latent.flatten(0, 1), 
            noise.flatten(0, 1), 
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))


        # Compute loss
        # flow_pred, x0_pred = self.generator(
        #     noisy_image_or_video=noisy_latents,
        #     conditional_dict=conditional_dict,
        #     timestep=timestep,
        #     clean_x=clean_latent_aug if self.teacher_forcing else None,
        #     aug_t=timestep_clean_aug if self.teacher_forcing else None
        # )
        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=None,
            aug_t=None
        )
        
        # 随机裁剪以节省显存（如果启用）
        if self.use_random_crop:
            flow_pred, training_target = self.random_crop(flow_pred, training_target)
        
        # loss = torch.nn.functional.mse_loss(flow_pred.float(), training_target.float())
        loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction='none'
        ).mean(dim=(2, 3, 4))
        loss = loss * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        loss = loss.mean()

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach()
        }
        return loss, log_dict
    