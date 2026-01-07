# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

try:
    from .attention import flash_attention
except:
    from attention import flash_attention

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


#@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000,
                scaling: str = "none",          # "none" | "ntk" | "yarn"
                factor: float = 8.0,            # 放大上下文的倍率，例如从4K到32K可用8.0
                yarn_alpha: float = 0.8,        # YARN平滑指数(0.5~1.0常用)
                yarn_short_factor: float = 1.0, # 近程段保真(=1表示近程不缩放)
                ):
    '''
    生成长度为 max_seq_len、维度为 dim 的复数频率表（极坐标形式）
    
    '''
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


#@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    '''
    先把最后一维一分为二（偶/奇），视作复数后做乘法（复乘就是二维旋转）
    
    '''
    n, c = x.size(2), x.size(3) // 2

    # split freqs 第一段用于 frame，后两段分别用于 height/width
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)  # 把 freqs 分成三段

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        # 把三个轴的频率按广播扩展到 [F,H,W,*] 并 concat 到最后一维，形成 [seq_len, 1, (df+dh+dw)]
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
    return torch.stack(output).type_as(x)


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
        return super().forward(x).type_as(x)


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

    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        
        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]: # 未初始化就计算一次
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else: # 已经初始化就使用缓存
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x




WAN_CROSSATTENTION_CLASSES = {
    'cross_attn': WanCrossAttention,
}

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
        #assert e.dtype == torch.float32
        #with torch.amp.autocast('cuda', dtype=torch.float32):
        #print(f"DEBUG: modulation shape={self.modulation.unsqueeze(0).shape}, e shape={e.shape}")
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2) # list 六个 [1, seq_len, 1, C]
        #assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        #with torch.amp.autocast('cuda', dtype=torch.float32):
        x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            #with torch.amp.autocast('cuda', dtype=torch.float32):
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
        #assert e.dtype == torch.float32
        #with torch.amp.autocast('cuda', dtype=torch.float32):
        #print("in head, self.modulation.shape, e.shape: ", self.modulation.shape, e.shape)
        
        e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x
    

class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class RegisterTokens(nn.Module):
    def __init__(self, num_registers: int, dim: int):
        super().__init__()
        self.register_tokens = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.rms_norm = WanRMSNorm(dim, eps=1e-6)

    def forward(self):
        return self.rms_norm(self.register_tokens)

    def reset_parameters(self):
        nn.init.normal_(self.register_tokens, std=0.02)
        
        

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
            nn.Conv3d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(16, mid_dim),
            nn.SiLU(),
            nn.Conv3d(mid_dim, model_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
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

        # dt-aware conditioning (initialized as no-op)
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
        
        # 5. Zero Initialization (零初始化)
        # 保证刚开始训练时，注入的特征全是 0，不影响主干
        for layer in self.zero_layers.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, lr_latents, guidance_t_emb, guidance_dt_emb=None):
        """
        lr_latents: [B, C, F, H, W]
        t_sinusoidal_emb: [B, freq_dim] <- 这是原始的正弦位置编码
        """
        # A. 提取特征  
        x = self.backbone(lr_latents)  # [B, C, T, H, ]
        x = x.flatten(2).transpose(1, 2) # [B, SeqLen, Dim]
        
        w = self.adapter_time_proj(guidance_t_emb)           # [B, D]
        w = w + self.adapter_dt_proj(guidance_dt_emb)
        scale = w                   # [B, D]
        scale = torch.tanh(scale) 
        
        x = self.feature_norm(x)
        
        # B. 注入 Guidance Timestep (控制强度)。 类似于把 guidance 加到 feature 上
        x = x * (1 + scale.unsqueeze(1)) # + shift.unsqueeze(1) # Scale 调制，或者 add 也可以
            
        # C. 生成每一层的控制特征  Generate Per-Layer Controls
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
            
            control_feat = controls.get(block_idx)
            if control_feat is None:
                return args

            x = args[0] # [B, L_x, Dim] (如果是 List 或者是 Tensor，WanModel 里中间层通常是 Tensor)

            B, L, D = x.shape
            Lc = control_feat.shape[1]
            
            if Lc < L:
                pad = control_feat.new_zeros(B, L - Lc, D)
                control_feat = torch.cat([control_feat, pad], dim=1)
            
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
            # t: [B]
            t_freq = sinusoidal_embedding_1d(self.freq_dim, t).type_as(x_in[0])  # [B, freq_dim]
        else:
            # t: [B, F] (per-frame); 取首帧代表，避免传入 [B, F] 形状破坏 adapter 的线性层
            t_freq = sinusoidal_embedding_1d(self.freq_dim, t[:, 0]).type_as(x_in[0])  # [B, freq_dim]

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
        
        #self._current_spatial_ctx = {'controls': controls}
        ctx = {'controls': controls, 'skip_masks': {}}

        self._current_spatial_ctx = ctx
        self._spatial_ctx_cache = ctx   # <- persists for checkpoint recompute

        try:
            kwargs['lr_latents'] = lr_latents
            if dt is not None:
                kwargs["dt"] = dt
            out = original_forward(x, t, context, seq_len, **kwargs)
        finally:
            # clear only when not using checkpointing; otherwise leave for replay
            if not self.gradient_checkpointing:
                self._current_spatial_ctx = None
                self._spatial_ctx_cache = None
        return out

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
    _supports_gradient_checkpointing = True

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

        assert model_type in ['t2v', 'i2v', 'ti2v']
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
            self.freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6), scaling="yarn", factor=1.5, yarn_alpha=0.8, yarn_short_factor=1.0),
                rope_params(1024, 2 * (d // 6), scaling="yarn", factor=1.5, yarn_alpha=0.8, yarn_short_factor=1.0)
            ], dim=1)
        else:
            self.freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ], dim=1)         
        

        # initialize weights
        self.init_weights()
        
        self.gradient_checkpointing = False

        # NOTE: dt_embedding is created post-load (see enable_dt_conditioning) to avoid
        # diffusers/accelerate meta-loading failures when pretrained weights don't include it.
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
        
    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value
        
    def forward(
        self,
        *args,
        **kwargs
    ):
        # if kwargs.get('classify_mode', False) is True:
        # kwargs.pop('classify_mode')
        # return self._forward_classify(*args, **kwargs)
        # else:
        return self._forward(*args, **kwargs)

    def _forward(
        self,
        x,
        t,
        context,
        seq_len,
        dt=None,
        classify_mode=False,
        concat_time_embeddings=False,
        register_tokens=None,
        cls_pred_branch=None,
        gan_ca_blocks=None,
        clip_fea=None,
        y=None,
        lr_latents=None,
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
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]  # list 1个 [1, 3072, t, h, w]
        
        
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]         # list 1个 [1, t*h*w, 3072]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)  # t*h*w
        
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)  # [B, seq_len]
        else:
            # support per-frame timestep: expand each frame value to its patch tokens, then pad/crop to seq_len
            if t.shape[1] != seq_len:
                num_frames = grid_sizes[0][0].item()
                frame_len = seq_lens[0].item() // num_frames  # tokens per frame (unpadded length)
                t_expanded = torch.repeat_interleave(t, frame_len, dim=1)  # [B, num_frames*frame_len]
                if t_expanded.shape[1] < seq_len:
                    pad = seq_len - t_expanded.shape[1]
                    t = torch.cat([t_expanded, t_expanded[:, -1:].expand(-1, pad)], dim=1)
                else:
                    t = t_expanded[:, :seq_len]
            # else: already [B, seq_len]
        bt = t.size(0)
        t = t.flatten()      # [B*seq_len]
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).type_as(x)) # [1, seqlen, 3072]

        if dt is not None:
            if self.dt_embedding is None:
                raise RuntimeError("dt was provided but dt conditioning is not enabled; call enable_dt_conditioning() after loading.")
            if dt.dim() == 1:
                dt = dt.expand(dt.size(0), seq_len)  # [B, seq_len]
            else:
                if dt.shape[1] != seq_len:
                    num_frames = grid_sizes[0][0].item()
                    frame_len = seq_lens[0].item() // num_frames
                    dt_expanded = torch.repeat_interleave(dt, frame_len, dim=1)  # [B, num_frames*frame_len]
                    if dt_expanded.shape[1] < seq_len:
                        pad = seq_len - dt_expanded.shape[1]
                        dt = torch.cat([dt_expanded, dt_expanded[:, -1:].expand(-1, pad)], dim=1)
                    else:
                        dt = dt_expanded[:, :seq_len]
            dt_flat = dt.flatten()
            e_dt = self.dt_embedding(
                sinusoidal_embedding_1d(self.freq_dim, dt_flat).unflatten(0, (bt, seq_len)).type_as(x)
            )
            e = e + e_dt
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))           # [1, 27280, 6, 3072]

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        common_kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        def create_custom_forward(module, **module_kwargs):
            def custom_forward(*inputs):
                inp_x, inp_lr_context = inputs
                return module(inp_x, lr_latents=inp_lr_context, **module_kwargs)
            return custom_forward

        # TODO: Tune the number of blocks for feature extraction
        final_x = None

        gan_idx = 0
        for ii, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, **common_kwargs),
                    x, 
                    lr_latents, 
                    use_reentrant=False,
                )
            else:
                x = block(x, lr_latents=lr_latents, **common_kwargs)
                
                #print(f"the shape in {ii}, x.shape: {x.shape}")
                
        
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        if classify_mode:
            return torch.stack(x), final_x

        return torch.stack(x)
        #return [u for u in x]
    

    def unpatchify(self, x, grid_sizes, c=None):
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

        c = self.out_dim if c is None else c
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
        
        
    def reinit_patch_embedding(self, new_in_dim: int, new_param_init: str = "copy"):
        """重新构造 `patch_embedding` 以适配更大的输入通道数。

        Args:
            new_in_dim (int): 新的输入通道数（如 96）。
            new_param_init (str): 对新增通道权重的初始化方式，支持 "copy" | "zero"。
        """
        old_in_dim = self.config.in_dim  # 旧的输入通道数（如 48）
        if new_in_dim <= old_in_dim:
            raise ValueError(
                f"new_in_dim({new_in_dim}) 必须大于旧的 in_dim({old_in_dim})")

        # 保存旧卷积权重与 bias
        old_weight = self.patch_embedding.weight.detach().clone()
        old_bias = self.patch_embedding.bias.detach().clone()

        # 构造新的卷积层，并保持 device / dtype 与旧权重一致
        new_conv = torch.nn.Conv3d(
            new_in_dim,
            old_weight.shape[0],
            kernel_size=self.patch_size,  # 与 __init__ 保持一致
            stride=self.patch_size,
            bias=True,
        ).to(old_weight.device, dtype=old_weight.dtype)

        with torch.no_grad():
            # 复制旧通道权重
            new_conv.weight[:, :old_in_dim] = old_weight

            # 处理新增通道
            extra_c = new_in_dim - old_in_dim
            if new_param_init == "zero":
                new_conv.weight[:, old_in_dim:] = 0.0
            elif new_param_init == "copy":
                # 复制模式要求新通道数必须是旧通道数的整数倍
                if extra_c % old_in_dim != 0:
                    raise ValueError(
                        f"In 'copy' mode, (new_in_dim({new_in_dim}) - old_in_dim({old_in_dim})) must be divisible by old_in_dim({old_in_dim})")
                # 将旧权重循环填充到新增通道
                repeat_times = extra_c // old_in_dim
                repeated = old_weight.repeat(1, repeat_times, 1, 1, 1)
                new_conv.weight[:, old_in_dim:] = repeated
            else:
                raise ValueError(f"Invalid new_param_init: {new_param_init}")
            new_conv.bias = torch.nn.Parameter(old_bias)

        # 替换并同步配置
        self.patch_embedding = new_conv
        self.config.in_dim = new_in_dim
        self.in_dim = new_in_dim
        
        
if __name__ == "__main__":
    
    
    
    model = WanModel_Upsample_Shortcut.from_pretrained(f"/mnt/vision-gen-ks3/ModelZoo/Video_Generation/Wan2.2-TI2V-5B", )
    model, lr_layers = register_spatial_control(model)
    
    model.eval()
    model.to("cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        lr_x = torch.randn(1, 48, 13, 16, 26, device="cuda", dtype=torch.bfloat16) # [B, F, C, H, W]
        
        
        noisy_image_or_video = torch.randn(1, 13, 48, 16, 26).to("cuda").to(dtype=torch.bfloat16)
        input_timestep = torch.randint(0, 1000, (1,)).to("cuda").to(dtype=torch.bfloat16)
        prompt_embeds = torch.randn(1, 512, 4096).to("cuda").to(dtype=torch.bfloat16)
        seq_len = 13*8*13
        
        flow_pred = model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),  # [b, c, t, h, w]
                    t=input_timestep, 
                    context=prompt_embeds,
                    seq_len=seq_len,
                    lr_latents=lr_x # 只需要这一路输入
                ).permute(0, 2, 1, 3, 4)
        
        print("flow_pred: ", flow_pred.shape)
