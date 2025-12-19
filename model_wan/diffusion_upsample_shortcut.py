from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from utils_long.wan2_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper2_2
from pipeline_long import SelfForcingTrainingPipeline


class SelfForcingWan_Upsample_SC(nn.Module):
    def __init__(self, args, device):
        """
        Initialize the Diffusion loss module.
        """
        super().__init__()
        self._initialize_models(args, device)
        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        
        
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
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
        self.timestep_shift = getattr(args, "timestep_shift", 5.0)
        self.teacher_forcing = getattr(args, "teacher_forcing", False)
        
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)
        
        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None
            
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)

      
    def _initialize_models(self, args, device):
        self.sr_mode = args.sr_mode
        self.seq_len = args.seq_len
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), seq_len=self.seq_len, sr=self.sr_mode)
        self.generator.model.requires_grad_(True)
        
        # 追加：冻结主干、仅保留 adapter
        if args.trainable_backbone is False:
            
            for name, p in self.generator.model.named_parameters():
                if not name.startswith("spatial_adapter"):
                    p.requires_grad_(False)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper2_2()
        self.vae.requires_grad_(False)
        
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    @staticmethod
    def _slice_batched_conditionals(conditional_dict: dict, mask: torch.Tensor) -> dict:
        out = {}
        for k, v in conditional_dict.items():
            if torch.is_tensor(v) and v.shape[:1] == mask.shape[:1]:
                out[k] = v[mask]
            else:
                out[k] = v
        return out

    def _call_generator(
        self,
        noisy_latents: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        lr_context,
        dt,
    ) -> torch.Tensor:
        if dt is None:
            return self.generator(
                noisy_image_or_video=noisy_latents,
                conditional_dict=conditional_dict,
                timestep=timestep,
                lr_context=lr_context,
            )

        try:
            return self.generator(
                noisy_image_or_video=noisy_latents,
                conditional_dict=conditional_dict,
                timestep=timestep,
                dt=dt,
                lr_context=lr_context,
            )
        except TypeError as e:
            if torch.is_tensor(dt) and torch.all(dt == 0):
                return self.generator(
                    noisy_image_or_video=noisy_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    lr_context=lr_context,
                )
            raise TypeError(
                "Shortcut training requires passing `dt` into the generator. "
                "Please update `utils_long/wan2_wrapper.py:WanDiffusionWrapper.forward(...)` and "
                "`wan2_long/modules/model_upsample.py:WanModel_Upsample` to accept/use `dt`."
            ) from e

    def _sample_dt_sigma(self, batch_size: int, device, dtype) -> torch.Tensor:
        """
        Sample discrete dt values for shortcut training.

        Non-zero dt candidates: {2^-7, 2^-6, ..., 2^-1, 1}.
        During training we uniformly sample from the first 7 values (exclude 1).
        """
        min_pow = int(getattr(self.args, "shortcut_min_dt_pow", 7))
        assert min_pow >= 1
        dt_candidates = torch.tensor(
            [2.0 ** (-p) for p in range(min_pow, 0, -1)],
            device=device,
            dtype=dtype,
        )  # [2^-min_pow, ..., 2^-1]
        dt = dt_candidates[torch.randint(0, dt_candidates.numel(), (batch_size,), device=device)]
        return dt # 对每个样本独立抽一个index

    def _sample_sc_timestep_and_mid(
        self,
        dt_sigma: torch.Tensor,  # [B_sc], float in sigma-space
        device,
        dtype=torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            timestep: [B_sc] timestep for x_t
            timestep_mid: [B_sc] timestep for x_{t - dt/2}
            dt_timestep: [B_sc] dt mapped to the same timestep scale
        """
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        timesteps = self.scheduler.timesteps.to(device=device, dtype=dtype)

        min_sigma = sigmas.min()
        max_sigma = sigmas.max()
        req_min = dt_sigma + min_sigma
        sigma_sample = req_min + torch.rand_like(req_min) * torch.clamp(max_sigma - req_min, min=0.0)

        idx = torch.argmin((sigmas[None, :] - sigma_sample[:, None]).abs(), dim=1)
        sigma = sigmas[idx]
        timestep = timesteps[idx]

        sigma_mid = torch.clamp(sigma - 0.5 * dt_sigma, min=min_sigma)
        idx_mid = torch.argmin((sigmas[None, :] - sigma_mid[:, None]).abs(), dim=1)
        timestep_mid = timesteps[idx_mid]

        dt_timestep = dt_sigma * float(self.num_train_timestep)
        return timestep, timestep_mid, dt_timestep


    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        clean_latent: torch.Tensor,
        unconditional_dict: dict = None,
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
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        """
        
        enable_shortcut = bool(getattr(self.args, "shortcut_enable", True))
        rate_sc = float(getattr(self.args, "shortcut_rate_sc", 0.25))
        w_sc = float(getattr(self.args, "shortcut_loss_sc_weight", 1.0))

        noise = torch.randn_like(clean_latent)
        #print(f"image_or_video_shape: {image_or_video_shape}")
        batch_size, num_frame = image_or_video_shape[:2]
        cond_latent = None
        cond_frames = 0
        if initial_latent is not None:
            cond_latent = initial_latent
            if cond_latent.dim() == 4:
                cond_latent = cond_latent.unsqueeze(1)
            cond_latent = cond_latent.to(device=self.device, dtype=self.dtype)
            cond_frames = cond_latent.shape[1]
            noise[:, :cond_frames] = 0
            

        # Randomly drop conditions so the model learns the unconditional branch for CFG.
        uncond_proba = getattr(self.args, "uncond_proba", 0.1)
        if uncond_proba > 0:
            mask = torch.bernoulli(
                torch.full((batch_size,), uncond_proba, device=self.device)
            ).bool()
            if mask.any():
                conditional_dict = dict(conditional_dict)
                prompt_cond = conditional_dict["prompt_embeds"].clone()
                prompt_uncond = unconditional_dict["prompt_embeds"]
                prompt_cond[mask] = prompt_uncond[mask]
                conditional_dict["prompt_embeds"] = prompt_cond
                
                if clean_latent_lr is not None:
                    clean_latent_lr = clean_latent_lr.clone()
                    clean_latent_lr[mask] = 0

        # ======== Sample (t, dt) for shortcut training ========
        # - Flow-matching branch uses dt=0
        # - Self-consistency branch uses dt in {2^-7,...,2^-1} (uniform)
        dt_sigma = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        sc_mask = torch.zeros(batch_size, device=self.device, dtype=torch.bool)   # 记录第几个样本走shortcut
        if enable_shortcut and rate_sc > 0:
            sc_mask = torch.rand(batch_size, device=self.device) < rate_sc
            if sc_mask.any():
                dt_sigma[sc_mask] = self._sample_dt_sigma(sc_mask.sum().item(), device=self.device, dtype=torch.float32)

        timesteps = self.scheduler.timesteps.to(device=self.device, dtype=self.dtype)
        # Flow-matching timestep sampling: uniform over scheduler grid
        timestep_base = torch.empty(batch_size, device=self.device, dtype=self.dtype)
        fm_mask = ~sc_mask
        if fm_mask.any():
            idx_fm = torch.randint(0, len(timesteps), (fm_mask.sum().item(),), device=self.device, dtype=torch.long)
            timestep_base[fm_mask] = timesteps[idx_fm]

        # 给 shortcut 样本采样 (t, t_mid, dt)
        timestep_mid_base = None
        dt_timestep = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        if sc_mask.any():
            t_sc, t_mid_sc, dt_sc = self._sample_sc_timestep_and_mid(dt_sigma[sc_mask], device=self.device)
            timestep_base[sc_mask] = t_sc.to(dtype=self.dtype)
            dt_timestep[sc_mask] = dt_sc
            timestep_mid_base = torch.empty(batch_size, device=self.device, dtype=self.dtype)
            timestep_mid_base[sc_mask] = t_mid_sc.to(dtype=self.dtype)

        timestep = timestep_base[:, None].expand(batch_size, num_frame)  # [B, F]
        dt = dt_timestep.to(dtype=self.dtype)[:, None].expand(batch_size, num_frame)  # [B, F]
        # if cond_frames > 0:
        #     timestep[:, :cond_frames] = 0
        # Flow Matching: x_t = (1-sigma) * x0 + sigma * noise
        if dist.get_rank() == 0:
            print(self.scheduler.timesteps)
        print(f"dt: {dt[:, 0]}")
        print(f"dt[sc_mask]: {dt[sc_mask][:, 0]}")
        print(f"timestep: {timestep[:, 0]}")
            
            
        exit()
        
        
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


        dt_in = dt if enable_shortcut else None
        flow_pred = self._call_generator(
            noisy_latents=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            dt=dt_in,
            lr_context=clean_latent_lr,
        )
        

        # losses
        
        # ===== 计算 Flow Matching Loss =======
        mse_fm = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction="none"
        ).mean(dim=(2, 3, 4))  # [B, F]

        weights_all = self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        if cond_frames > 0:
            weights_all[:, :cond_frames] = 0

        weights_fm = weights_all.clone()
        if enable_shortcut and sc_mask.any():
            weights_fm[sc_mask] = 0
        denom_fm = torch.clamp(weights_fm.sum(), min=1e-8)
        loss_fm = torch.sum(mse_fm * weights_fm) / denom_fm
        
        
        

        loss_sc = torch.zeros((), device=self.device, dtype=torch.float32)
        if enable_shortcut and sc_mask.any():
            # self-consistency targets:
            # v_sc(x_t, t, dt) ≈ ( v(x_t, t, dt/2) + v(x_{t-dt/2}, t-dt/2, dt/2) ) / 2
            x_sc = noisy_latents[sc_mask]
            t_sc_full = timestep[sc_mask]
            dt_sc_full = dt[sc_mask]
            dt_sc_half = dt_sc_full * 0.5
            dt_sc_half_sigma = dt_sigma[sc_mask] * 0.5

            if timestep_mid_base is None:
                raise RuntimeError("Internal error: missing timestep_mid for self-consistency samples.")
            t_sc_mid = timestep_mid_base[sc_mask][:, None].expand(-1, num_frame)
            conditional_dict_sc = self._slice_batched_conditionals(conditional_dict, sc_mask)
            lr_context_sc = clean_latent_lr[sc_mask] if clean_latent_lr is not None else None

            with torch.no_grad():
                v1 = self._call_generator(
                    noisy_latents=x_sc,
                    conditional_dict=conditional_dict_sc,
                    timestep=t_sc_full,
                    dt=dt_sc_half,
                    lr_context=lr_context_sc,
                )
                x_mid = x_sc - dt_sc_half_sigma[:, None, None, None, None] * v1
                v2 = self._call_generator(
                    noisy_latents=x_mid,
                    conditional_dict=conditional_dict_sc,
                    timestep=t_sc_mid,
                    dt=dt_sc_half,
                    lr_context=lr_context_sc,
                )
                v_sc = 0.5 * (v1 + v2)

            mse_sc = torch.nn.functional.mse_loss(
                flow_pred[sc_mask].float(), v_sc.float(), reduction="none"
            ).mean(dim=(2, 3, 4))  # [B_sc, F]

            weights_sc = weights_all.clone()
            weights_sc[~sc_mask] = 0
            denom_sc = torch.clamp(weights_sc.sum(), min=1e-8)
            loss_sc = torch.sum(mse_sc * weights_sc[sc_mask]) / denom_sc

        loss = loss_fm + w_sc * loss_sc

        log_dict = {
            "x0": clean_latent.detach(),
            "loss_fm": loss_fm.detach(),
            "loss_sc": loss_sc.detach(),
        }
        return loss, log_dict
