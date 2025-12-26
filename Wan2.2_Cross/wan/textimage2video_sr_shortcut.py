# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model_upsample_shortcut import WanModel_Upsample_Shortcut
from .modules.model_upsample_shortcut2 import WanModel_Upsample_Shortcut2
from .modules.t5 import T5EncoderModel
from .modules.vae2_2 import Wan2_2_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import best_output_size, masks_like


class WanTI2V_Upsample_Shortcut:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        wan_ckpt=None,
        use_ema=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        from .modules.model_upsample_shortcut2 import register_spatial_control
        self.model = WanModel_Upsample_Shortcut2.from_pretrained(checkpoint_dir)
        # from .modules.model_upsample_shortcut import register_spatial_control
        # self.model = WanModel_Upsample_Shortcut.from_pretrained(checkpoint_dir)
        self.model.enable_dt_conditioning()
        #==============load the model from the checkpoint (在 FSDP 之前加载)=============
        adapter_state_dict = None
        if wan_ckpt is not None:
            if not use_sp:
                print(f"Loading Wan model from {wan_ckpt}")
                state_dict = torch.load(wan_ckpt, map_location="cpu")
            else:
                print(f"Loading Wan model from {wan_ckpt} for SP mode")
                state_dict = None  # 先占位
                if not dist.is_initialized() or dist.get_rank() == 0:
                    state_dict = torch.load(wan_ckpt, map_location="cpu")
                    print(f"[rank0] loaded Wan model from {wan_ckpt}")

                if dist.is_initialized():
                    dist.barrier()  # 确保其它 rank 等待
                    obj_list = [state_dict]
                    dist.broadcast_object_list(obj_list, src=0)
                    state_dict = obj_list[0]  # 其它 rank 拿到同一个 state_dict

            if use_ema:
                print("------- Using EMA Weight ------")
                generator_state_dict = state_dict["generator_ema"]
            else:
                print("------- Using None EMA Weight ------")
                generator_state_dict = state_dict["generator"]

            # 先去掉 FSDP 产生的中间前缀
            def clean_fsdp_keys(d):
                new = {}
                for k, v in d.items():
                    k2 = (
                        k.replace("_fsdp_wrapped_module.", "")
                        .replace("_checkpoint_wrapped_module.", "")
                        .replace("_orig_mod.", "")
                    )
                    new[k2] = v
                return new

            if use_ema:
                generator_state_dict = clean_fsdp_keys(generator_state_dict)

            def strip_prefix(d, prefix="model."):
                if all(k.startswith(prefix) for k in d.keys()):
                    return {k[len(prefix):]: v for k, v in d.items()}
                return d

            generator_state_dict = strip_prefix(generator_state_dict)

            # 分离 adapter 的权重
            adapter_keys = [k for k in generator_state_dict.keys() if k.startswith("spatial_adapter.")]
            adapter_state_dict = {k[len("spatial_adapter."):]: generator_state_dict.pop(k) for k in adapter_keys}
            print(f"Found {len(adapter_keys)} adapter keys in checkpoint")

            # 加载主模型权重
            missing_keys, unexpected_keys = self.model.load_state_dict(generator_state_dict)
            print(f"Missing keys: {len(missing_keys)}, unexpected: {len(unexpected_keys)}")
            print(missing_keys)
        # ==============================================================
 
        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        
        # ======================================================================
        self.model, _ = register_spatial_control(self.model)

        if adapter_state_dict:
            try:
                missing_keys, unexpected_keys = self.model.spatial_adapter.load_state_dict(
                    adapter_state_dict, strict=False
                )
                print(
                    f"Successfully loaded spatial_adapter weights (missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})"
                )
            except Exception as e:
                print(f"Warning: Failed to load spatial_adapter weights: {e}")
        else:
            print("No spatial_adapter weights found in checkpoint")
        # ======================================================================
         
         
        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _shortcut_quantize_dt(self, t_cur: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
        """
        Quantize inference-time dt to match the training-time shortcut dt distribution.

        Training uses dt candidates derived from a (snapped) timestep scale T:
          dt_candidates = {T, floor(T/2), ..., floor(T/2^k)}  (k=shortcut_min_dt_pow)
        and dt is made even so dt/2 is an integer (used in the self-consistency midpoint).

        Here we:
          1) optionally snap `t_cur` onto anchor timesteps (e.g. 600/700/...)
          2) build dt_candidates from that snapped T
          3) snap the raw solver step |t_cur - t_next| onto the nearest candidate.
        """
        device = t_cur.device
        t_cur_f = t_cur.to(dtype=torch.float32)
        t_next_f = t_next.to(dtype=torch.float32)
        dt_raw = (t_cur_f - t_next_f).abs()

        k = int(getattr(self.config, "shortcut_min_dt_pow", 7))
        t_min = float(getattr(self.config, "shortcut_t_min", 600))
        t_max = float(getattr(self.config, "shortcut_t_max", 1000))
        t_stride = float(getattr(self.config, "shortcut_t_stride", 100))
        snap_t = bool(getattr(self.config, "shortcut_infer_snap_t", True))

        if snap_t:
            anchors = torch.arange(t_min, t_max + 1e-6, t_stride, device=device, dtype=torch.float32)
            T = anchors[torch.argmin((anchors - t_cur_f).abs())]
        else:
            T = t_cur_f

        powers = torch.arange(0, k + 1, device=device, dtype=torch.long)
        div = (2**powers).to(dtype=torch.float32)
        dt_cands = torch.div(T, div, rounding_mode="floor").to(dtype=torch.int64)  # [K]

        # ensure dt/2 is an integer-like jump and keep dt >= 2
        dt_cands = torch.clamp(dt_cands, min=2)
        dt_cands = dt_cands - (dt_cands % 2)
        dt_cands = torch.clamp(dt_cands, min=2)

        dt_idx = dt_cands[torch.argmin((dt_cands.to(torch.float32) - dt_raw).abs())]
        return dt_idx.to(dtype=torch.int64)

    def generate(self,
                 input_prompt,
                 cond_latent=None,
                 img=None,
                 size=(1280, 704),
                 max_area=704 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            cond_latent (`torch.Tensor`, *optional*, defaults to None):
                Conditional latent tensor. Shape: [B, T, C, H, W]
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # i2v
        if img is not None:
            return self.i2v(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model)
        # t2v
        
        print('input_size:', size)
        print('max_area:', max_area)
        
        return self.t2v(
            input_prompt=input_prompt,
            cond_latent=cond_latent,
            size=size,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model)

    def t2v(self,
            input_prompt,
            cond_latent,
            size=(1280, 704),
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            cond_latent (`torch.Tensor`, *optional*, defaults to None):
                Conditional latent tensor. Shape: [B, C, T, H, W]
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size
        
        print('target_shape:', target_shape)  # [48, 31, 90, 160]
        print('seq_len:', seq_len)  # 121

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        
        # 先处理 cond_latent 的维度（如果有的话）
        if cond_latent is not None:
            # if cond_latent.dim() == 5:  # [B, C, T, H, W]
            #     cond_latent = cond_latent.squeeze(0)  # 变成 [C, T, H, W]
            print(f'cond_latent_shape: {cond_latent.shape}')
        
        # 暂时先生成纯噪声（后面会根据 scheduler 的第一个 sigma 重新初始化）
        noise = [torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g)
        ]
        

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
                print(f'Timesteps: {len(timesteps)} steps, range [{timesteps[0]:.1f}, {timesteps[-1]:.1f}]')
                
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")


            latents = noise
            print(f'✓ T2V Mode: pure noise, mean={latents[0].mean():.4f}, std={latents[0].std():.4f}')
            
            mask1, mask2 = masks_like(latents, zero=False)
            
            
            # ============ LR Context ============
            #cond_latent = cond_latent.permute(0, 2, 1, 3, 4).contiguous()   # [B, C, T, h, w]

            arg_c = {'context': context, 'seq_len': seq_len, 'lr_latents':cond_latent}
            #arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len, 'lr_latents':cond_latent}

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()
            
            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = torch.stack([t])
                t_next = timesteps[i + 1] if (i + 1) < len(timesteps) else timestep.new_zeros(())
                dt_idx = self._shortcut_quantize_dt(timestep[0], t_next)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                dt_embed = (mask2[0][0][:, ::2, ::2] * dt_idx).flatten()
                dt_embed = torch.cat([
                    dt_embed,
                    dt_embed.new_ones(seq_len - dt_embed.size(0)) * dt_idx
                ])
                dt_embed = dt_embed.unsqueeze(0)

                noise_pred_cond = self.model(latent_model_input, t=timestep, dt=dt_embed, **arg_c)[0]
                noise_pred_uncond = self.model(latent_model_input, t=timestep, dt=dt_embed, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                #noise_pred = noise_pred_cond
                
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
                
                # 每 10 步打印一次统计信息
                if i % 10 == 0 or i == len(timesteps) - 1:
                    print(f'  Step {i}/{len(timesteps)}, t={t:.1f}, '
                          f'latent: mean={latents[0].mean():.4f}, std={latents[0].std():.4f}')
            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            # if self.rank == 0:
            #     videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        #return videos[0] if self.rank == 0 else None
        
        return x0

    def i2v(self,
            input_prompt,
            img,
            max_area=704 * 1280,
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=40,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (121)
                - H: Frame height (from max_area)
                - W: Frame width (from max_area)
        """
        # preprocess
        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        z = self.vae.encode([img])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise
            mask1, mask2 = masks_like([noise], zero=True)
            latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

            arg_c = {
                'context': [context[0]],
                'seq_len': seq_len,
            }

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([t]).to(self.device)
                t_next = timesteps[i + 1] if (i + 1) < len(timesteps) else timestep.new_zeros(())
                dt_idx = self._shortcut_quantize_dt(timestep[0], t_next)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                dt_embed = (mask2[0][0][:, ::2, ::2] * dt_idx).flatten()
                dt_embed = torch.cat([
                    dt_embed,
                    dt_embed.new_ones(seq_len - dt_embed.size(0)) * dt_idx
                ])
                dt_embed = dt_embed.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, dt=dt_embed, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, dt=dt_embed, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
