from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn
from utils_long.wan2_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper2_2
from pipeline_long import SelfForcingTrainingPipeline


class SelfForcingWan_Upsample_Causal(nn.Module):
    def __init__(self, args, device):
        """
        Initialize the Diffusion loss module.
        """
        super().__init__()
        self._initialize_models(args, device)
        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        
        
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 6)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        
        
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
        self.sr_mode = args.sr_mode
        self.seq_len = args.seq_len
        self.causal=args.causal
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), seq_len=self.seq_len, sr=self.sr_mode, causal=self.causal)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper2_2()
        self.vae.requires_grad_(False)
        
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)


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
        #print(f"image_or_video_shape: {image_or_video_shape}")
        batch_size,  num_frame = image_or_video_shape[:2]
        cond_latent = None
        cond_frames = 0
        if initial_latent is not None:
            cond_latent = initial_latent
            if cond_latent.dim() == 4:
                cond_latent = cond_latent.unsqueeze(1)
            cond_latent = cond_latent.to(device=self.device, dtype=self.dtype)
            cond_frames = cond_latent.shape[1]
            noise[:, :cond_frames] = 0

        # Step 2: Randomly sample a timestep and add noise to denoiser inputs (Flow Matching)
        # 从[0, 1000)中随机采样timestep index
        index = torch.randint(
            0,
            len(self.scheduler.timesteps),
            (batch_size, num_frame),
            device=self.device,
            dtype=torch.long
        )
        if self.num_frame_per_block > 1:
            if self.independent_first_frame:
                # 第一帧单独，其余帧按块共享
                idx_rest = index[:, 1:].reshape(batch_size, -1, self.num_frame_per_block)
                idx_rest[:, :, 1:] = idx_rest[:, :, 0:1]
                index = torch.cat([index[:, :1], idx_rest.reshape(batch_size, -1)], dim=1)
            else:
                idx_block = index.reshape(batch_size, -1, self.num_frame_per_block)
                idx_block[:, :, 1:] = idx_block[:, :, 0:1]
                index = idx_block.reshape(batch_size, -1)

        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)  # [B, F]
 
        # if cond_frames > 0:
        #     timestep[:, :cond_frames] = 0
        # Flow Matching: x_t = (1-sigma) * x0 + sigma * noise
        
        
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
        
        if cond_frames > 0 and cond_latent.shape[2:] == noisy_latents.shape[2:]:
            noisy_latents[:, :cond_frames] = cond_latent


        flow_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            lr_context=clean_latent_lr,
        )
        

        
        # loss = torch.nn.functional.mse_loss(flow_pred.float(), training_target.float())
        loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction='none'
        ).mean(dim=(2, 3, 4))
        weights = self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        if cond_frames > 0:
            weights[:, :cond_frames] = 0
        denom = torch.clamp(weights.sum(), min=1e-8)
        loss = torch.sum(loss * weights) / denom

        log_dict = {
            "x0": clean_latent.detach(),
            #"x0_pred": x0_pred.detach()
        }
        return loss, log_dict