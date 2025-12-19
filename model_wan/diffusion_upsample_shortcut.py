import math
from typing import Tuple
import torch
from torch import nn
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

    def _dt_idx_candidates(self, num_steps: int, device) -> torch.Tensor:
        """
        Build discrete dt candidates on the scheduler grid index scale.

        We start from a minimum dt of 2^-k (k=shortcut_min_dt_pow, default 7),
        then repeatedly double until 2^-1, giving 7 candidates.

        For num_train_timestep=1000, k=7 => dt_idx_min ~= round(1000 * 2^-7) ~= 8
        candidates: {8, 16, 32, 64, 128, 256, 512}.
        """
        min_pow = int(getattr(self.args, "shortcut_min_dt_pow", 7))
        assert min_pow >= 1

        dt_min = int(round(float(self.num_train_timestep) * (2.0 ** (-min_pow))))
        dt_min = max(2, dt_min)  # ensure dt/2 >= 1
        dt_min = 1 << int(round(math.log2(dt_min)))  # snap to power-of-two for exact doubling
        dt_min = max(2, dt_min)

        cands = [dt_min * (2**i) for i in range(min_pow)]  # 2^-k ... 2^-1
        cands = [c for c in cands if c < num_steps]
        cands = sorted(set(cands))
        if not cands:
            raise ValueError(f"No valid dt_idx candidates for num_steps={num_steps}, shortcut_min_dt_pow={min_pow}.")

        # Ensure even dt_idx so dt/2 is an integer grid jump.
        cands = [c if (c % 2 == 0) else (c + 1) for c in cands]
        cands = [c for c in cands if c < num_steps]
        cands = sorted(set(cands))
        if not cands:
            raise ValueError(f"No valid even dt_idx candidates for num_steps={num_steps}, shortcut_min_dt_pow={min_pow}.")

        return torch.tensor(cands, device=device, dtype=torch.long)


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

        # ======== Sample (t_idx, dt_idx) on the scheduler grid ========
        # We sample indices for perfect consistency:
        # - scheduler.add_noise / training_weight uses argmin over the same grid
        # - the network time embedding encodes the exact grid value timesteps[t_idx]
        # - shortcut step size is conditioned by an integer dt_idx ("how many grid steps to jump")
        num_steps = len(self.scheduler.timesteps)
        timesteps_grid = self.scheduler.timesteps.to(device=self.device, dtype=self.dtype)  # [num_steps], float
        sigmas_grid = self.scheduler.sigmas.to(device=self.device, dtype=torch.float32)     # [num_steps], float

        sc_mask = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        if enable_shortcut and rate_sc > 0:
            sc_mask = torch.rand(batch_size, device=self.device) < rate_sc

        dt_idx = torch.zeros(batch_size, device=self.device, dtype=torch.long)  # 0 for FM, >0 for SC
        if enable_shortcut and sc_mask.any():
            dt_candidates = self._dt_idx_candidates(num_steps=num_steps, device=self.device)
            dt_idx_sc = dt_candidates[torch.randint(0, dt_candidates.numel(), (sc_mask.sum().item(),), device=self.device)]
            dt_idx[sc_mask] = dt_idx_sc

        t_idx = torch.empty(batch_size, device=self.device, dtype=torch.long)
        fm_mask = ~sc_mask
        if fm_mask.any():
            t_idx[fm_mask] = torch.randint(0, num_steps, (fm_mask.sum().item(),), device=self.device, dtype=torch.long)
        if sc_mask.any():
            max_start = (num_steps - 1) - dt_idx[sc_mask]  # ensure t_idx + dt_idx <= num_steps - 1
            max_start = torch.clamp(max_start, min=0)
            t_idx_sc = (torch.rand_like(max_start.float()) * (max_start.float() + 1.0)).floor().to(torch.long)
            t_idx[sc_mask] = t_idx_sc

        t_mid_idx = t_idx + (dt_idx // 2)

        timestep_base = timesteps_grid[t_idx]  # [B]
        timestep = timestep_base[:, None].expand(batch_size, num_frame)  # [B, F]

        # dt conditioning uses integer grid step counts (encoded as numbers, but semantically dt_idx)
        dt = dt_idx.to(dtype=self.dtype)[:, None].expand(batch_size, num_frame)  # [B, F]
        
        
        print(f'dt: {dt[:, 0]}')
        print(f'timestep_base: {timestep_base[:, 0]}')
        exit()
        
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
            # v_sc(x_t, t, dt_idx) ≈ ( v(x_t, t, dt_idx/2) + v(x_mid, t_mid, dt_idx/2) ) / 2
            # where x_mid is advanced by the *actual* grid delta in sigma space.
            x_sc = noisy_latents[sc_mask]
            t_sc_full = timestep[sc_mask]

            dt_idx_sc = dt_idx[sc_mask]
            t_idx_sc = t_idx[sc_mask]
            t_mid_idx_sc = t_mid_idx[sc_mask]

            dt_half = (dt_idx_sc // 2).to(dtype=self.dtype)[:, None].expand(-1, num_frame)
            t_sc_mid = timesteps_grid[t_mid_idx_sc].to(dtype=self.dtype)[:, None].expand(-1, num_frame)

            sigma_sc = sigmas_grid[t_idx_sc]
            sigma_mid_sc = sigmas_grid[t_mid_idx_sc]
            delta_sigma_half = (sigma_mid_sc - sigma_sc).to(dtype=x_sc.dtype)  # [B_sc], usually negative

            conditional_dict_sc = self._slice_batched_conditionals(conditional_dict, sc_mask)
            lr_context_sc = clean_latent_lr[sc_mask] if clean_latent_lr is not None else None

            with torch.no_grad():
                v1 = self._call_generator(
                    noisy_latents=x_sc,
                    conditional_dict=conditional_dict_sc,
                    timestep=t_sc_full,
                    dt=dt_half,
                    lr_context=lr_context_sc,
                )
                x_mid = x_sc + delta_sigma_half[:, None, None, None, None] * v1
                v2 = self._call_generator(
                    noisy_latents=x_mid,
                    conditional_dict=conditional_dict_sc,
                    timestep=t_sc_mid,
                    dt=dt_half,
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
