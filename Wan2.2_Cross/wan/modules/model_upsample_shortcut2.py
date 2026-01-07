# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel_Upsample_Shortcut']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000,
                scaling: str = "none",          # "none" | "ntk" | "yarn"
                factor: float = 8.0,            # 放大上下文的倍率，例如从4K到32K可用8.0
                yarn_alpha: float = 0.8,        # YARN平滑指数(0.5~1.0常用)
                yarn_short_factor: float = 1.0, # 近程段保真(=1表示近程不缩放)
                ):
    assert dim % 2 == 0
    # ===== 原始频率计算 ===== theta^{-2i/d} 
    idx = torch.arange(0, dim, 2).to(torch.float64)
    base = 1.0 / torch.pow(theta, idx.div(dim))
    
    if scaling == "yarn" and factor != 1.0:
        # yarn: 高频压缩，低频保持
        t = torch.linspace(0.0, 1.0, base.numel(), dtype=torch.float64)
        half = base.numel() // 2
        yarn_scale = torch.ones_like(base)
        if half > 0:
            yarn_scale[:half] = (factor ** (t[:half] ** yarn_alpha))**(-1) * yarn_short_factor
        yarn_scale[half:] = (factor ** (t[half:] ** yarn_alpha))**(-1)
        base = base * yarn_scale
    
    # ---- 相位矩阵 ----
    freqs = torch.outer(torch.arange(max_seq_len), base)  # [max_seq_len, dim//2]
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        lr_latents=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
                       
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


# [新增] Scale Adapter 模块
class WanSpatialControlAdapter(nn.Module):
    def __init__(self, 
                 in_dim,          # LR Latent Channels (e.g., 16)
                 model_dim,       # Transformer Hidden Dim (e.g., 1536)
                 patch_size,      # (1, 2, 2)
                 num_blocks,      # 主干网络的层数，我们需要为每一层准备一个 ZeroLayer
                 control_block_indices=(1,),  # 实际需要注入的 block 索引
                 freq_dim=256 # 你的 guidance timestep 维度
                 ):
        super().__init__()
        self.model_dim = model_dim
        self.num_blocks = num_blocks
        self.freq_dim = freq_dim
        self.control_block_indices = tuple(int(i) for i in control_block_indices)
        
        # 1. 特征提取器 (简单的 3D CNN 提取结构)
        mid_dim = model_dim // 4
        self.backbone = nn.Sequential(
            nn.Conv3d(in_dim, mid_dim, kernel_size=patch_size, stride=patch_size),
            nn.GroupNorm(16, mid_dim),
            nn.SiLU(),
            nn.Conv3d(mid_dim, model_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            # 这里可以加深网络，或者用 ResNet Block
            
        )
        # --- 2. Feature Normalization (关键) ---
        # 在 Flatten 之后、进入 ZeroLinear 之前，做一个 LayerNorm
        # 确保输入给 ZeroLayers 的特征是标准分布的
        self.feature_norm = nn.LayerNorm(model_dim, eps=1e-6)

        # 3. Guidance Timestep Embedding (控制强度的开关)
        self.adapter_time_proj = nn.Sequential(
            nn.Linear(freq_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        
        # 3. Guidance Timestep Embedding (控制强度的开关)
        self.adapter_dt_proj = nn.Sequential(
            nn.Linear(freq_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        nn.init.zeros_(self.adapter_dt_proj[-1].weight)
        nn.init.zeros_(self.adapter_dt_proj[-1].bias)

        # 4. [核心] Per-Block Zero Layers
        # 只为需要注入的 block 创建零初始化线性层，避免无用参数
        self.zero_layers = nn.ModuleDict({
            str(i): nn.Linear(model_dim, model_dim) for i in self.control_block_indices
        })
        

    def forward(self, lr_latents, guidance_t_emb, guidance_dt_emb=None):
        """
        lr_latents: [B, C, F, H, W]
        t_sinusoidal_emb: [B, freq_dim] <- 这是原始的正弦位置编码
        """
        # A. 提取特征
        #print(lr_latents.shape)
        
        x = self.backbone(lr_latents)  # [B, C, T, H, ]
        x = x.flatten(2).transpose(1, 2) # [B, SeqLen, Dim]
        
        w = self.adapter_time_proj(guidance_t_emb)
        w = w + self.adapter_dt_proj(guidance_dt_emb)
        scale = w         # [B, D], [B, D]
        scale = torch.tanh(scale) 
        
        x = self.feature_norm(x)
        
        # B. 注入 Guidance Timestep (控制强度)
        # 类似于把 guidance 加到 feature 上
        # print(guidance_t_emb.shape) # guidance_t_emb: [B, Dim]       
        
        x = x * (1 + scale.unsqueeze(1))
            
        # C. 只生成需要注入的控制特征（按 block_idx）
        controls = {int(i): self.zero_layers[str(i)](x) for i in self.control_block_indices}
        return controls
    
def register_spatial_control(model):
    # 1. 实例化 Adapter
    adapter = WanSpatialControlAdapter(
        in_dim=model.in_dim,
        model_dim=model.dim,
        patch_size=model.patch_size,
        num_blocks=len(model.blocks),
        control_block_indices=getattr(model, "spatial_control_blocks", (1,)),
        freq_dim=model.freq_dim
    ).to(model.patch_embedding.weight.device)
    
    model.spatial_adapter = adapter
    
    # 存储 Hook 的 handle，方便后续清理
    model._spatial_hooks = []

    # =======================================================
    # 定义 Hook 函数工厂
    # =======================================================
    def create_block_hook(block_idx):
        def pre_forward_hook(module, args):
            """
            args 是一个 tuple: (x, e, seq_lens, ...)
            我们需要修改其中的 x (args[0])
            """
            # 1. 检查是否有 Control 上下文
            if not hasattr(model, '_current_spatial_ctx') or model._current_spatial_ctx is None:
                return args # 不做任何修改
            
            ctx = getattr(model, "_current_spatial_ctx", None)
            if ctx is None:
                ctx = getattr(model, "_spatial_ctx_cache", None)
            if ctx is None:
                return args
            
            controls = ctx.get('controls')
            
            if controls is None:
                return args
            
            # 只在指定的 block 注入
            control_feat = controls.get(block_idx)
            if control_feat is None:
                return args
            
            x = args[0] # [B, L_x, Dim] (如果是 List 或者是 Tensor，WanModel 里中间层通常是 Tensor)
            seq_lens = args[2]  # [B] (unpadded token length)
            
            # --- FIX STARTS HERE: Handle Sequence Parallelism Slicing ---
            # If input x is smaller than control, we assume SP is active and slice control
            if x.shape[1] != control_feat.shape[1]:
                if torch.distributed.is_initialized():
                    rank = torch.distributed.get_rank()
                    # Calculate the slice for this rank
                    # We assume the sequence is split evenly across the SP group (world_size)
                    local_len = x.shape[1]
                    start_idx = rank * local_len
                    end_idx = start_idx + local_len
                    
                    # Slice the global control feature to match local x
                    control_feat = control_feat[:, start_idx:end_idx, :]
                else:
                    if getattr(model, "spatial_control_strict_alignment", False):
                        raise RuntimeError(
                            f"Spatial control token length mismatch: x={x.shape} vs control={control_feat.shape}. "
                            "Padding/truncation is unsafe; ensure control uses the same tokenization as x."
                        )
                    return args
            # -----------------------------------------------------------
            
            # 4. [关键] 特征相加 (Feature Injection)
            # print(x.shape)
            x_new = x + (control_feat.type_as(x))
        
            # 5. 重新打包 args
            # Tuple 是不可变的，所以要新建一个
            new_args = (x_new,) + args[1:]
            return new_args
            
        return pre_forward_hook

    # =======================================================
    # 注册到每一层 Block
    # =======================================================
    # 先清理旧 hook
    if hasattr(model, '_spatial_hooks'):
        for h in model._spatial_hooks: h.remove()
    model._spatial_hooks = []

    for i, block in enumerate(model.blocks):
        # 为第 i 层注册 hook
        h = block.register_forward_pre_hook(create_block_hook(i))
        model._spatial_hooks.append(h)

    # =======================================================
    # 封装 Forward
    # =======================================================
    original_forward = model.forward

    def forward_with_spatial_control(self, x, t, context, seq_len, lr_latents=None, dt=None, **kwargs):
        # 处理 I2V 的拼接逻辑 (如果原模型有)
        x_in = x
        if kwargs.get('y') is not None:
             x_in = [torch.cat([u, v], dim=0) for u, v in zip(x, kwargs['y'])]
        
        # 1. 计算基础的 Sinusoidal Embedding (公用)
        # 这段逻辑是从原模型里提取出来的，为了让 Adapter 复用
        if t.dim() == 1:
            # 这里的 t 是 [Batch]
            # 扩展到 sequence 维度虽然是 WanModel 内部做的，
            # 但为了 Adapter，我们只需要 [Batch, FreqDim] 的 embedding 即可
            t_freq = sinusoidal_embedding_1d(self.freq_dim, t).type_as(x_in[0]) # [B, freq_dim]
        elif t.dim() == 2 and t.shape[1] != self.freq_dim:
            # 识别出这是被 pipeline 扩展过的 [1, SeqLen] 大张量
            # 我们只需要取第一个值作为全局时间步
            t_input = t[:, 0] 
            # 重新生成正确的 [1, 256] Embedding
            t_freq = sinusoidal_embedding_1d(self.freq_dim, t_input).type_as(x_in[0])
        
        else:
            # 如果 t 已经是 embedding (极少情况)
            t_freq = t
            
        dt_in = dt if dt is not None else kwargs.get("dt", None)
        dt_freq = None
        if dt_in is not None:
            if dt_in.dim() == 1:
                dt_freq = sinusoidal_embedding_1d(self.freq_dim, dt_in).type_as(x_in[0])  # [B, freq_dim]
            else:
                dt_freq = sinusoidal_embedding_1d(self.freq_dim, dt_in[:, 0]).type_as(x_in[0])  # [B, freq_dim]


        # 2. 运行 Adapter 
        # 将 LR 和 公用的 Time Freq 传入 Adapter
        # Adapter 内部会用自己的 MLP 处理这个 t_freq
        controls = self.spatial_adapter(lr_latents, t_freq, dt_freq)
        self._current_spatial_ctx = {'controls': controls}


        try:
            # 3. 调用原始 forward
            # 注意：原始 forward 内部还会算一遍 time embedding，
            # 虽然有点重复计算，但为了不魔改 _forward 内部代码，这是最干净的写法。
            # 只要 t 没变，逻辑就是一致的。
            kwargs['lr_latents'] = lr_latents
            kwargs["dt"] = dt_in
            return original_forward(x, t, context, seq_len, **kwargs)
        finally:
            self._current_spatial_ctx = None

    import types
    model.forward = types.MethodType(forward_with_spatial_control, model)
    
    return model, model.spatial_adapter




class WanModel_Upsample_Shortcut2(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        
        self.rope_scaling = None
        if self.rope_scaling == "yarn":
            print("========= Using yarn rope scaling ==========")
            self.freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                #rope_params(1024, d - 4 * (d // 6), scaling="yarn", factor=2, yarn_alpha=0.75, yarn_short_factor=1.0),
                rope_params(1024, 2 * (d // 6),
                            scaling="yarn", factor=1.5, yarn_alpha=0.8, yarn_short_factor=1.0),
                rope_params(1024, 2 * (d // 6),
                            scaling="yarn", factor=1.5, yarn_alpha=0.8, yarn_short_factor=1.0)
            ], dim=1)
        else:
            print("========= Using none rope scaling ==========")
            self.freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ], dim=1)    

        # initialize weights
        self.init_weights()
        
        
        self.dt_embedding = None
        
    def enable_dt_conditioning(self) -> None:
        """
        Create dt embedding parameters (for shortcut step-size conditioning).
        This must be called after loading pretrained weights.
        """
        if self.dt_embedding is not None:
            return

        dt_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim)
        )
        nn.init.zeros_(dt_embedding[-1].weight)
        nn.init.zeros_(dt_embedding[-1].bias)

        ref = self.time_embedding[0].weight
        self.dt_embedding = dt_embedding.to(device=ref.device, dtype=ref.dtype)

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        lr_latents=None,
        y=None,
        dt=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings & 步长 dt embeddings
        if self.dt_embedding is None:
            raise RuntimeError("dt was provided but dt conditioning is not enabled; call enable_dt_conditioning() after loading.")
        
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        if dt.dim() == 1:
            dt = dt.expand(dt.size(0), seq_len)  # [B, seq_len]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            
            t = t.flatten()
            dt = dt.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e_dt = self.dt_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        dt).unflatten(0, (bt, seq_len)).float())
            e = e + e_dt
            
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        
        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            lr_latents=lr_latents,
        )

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
