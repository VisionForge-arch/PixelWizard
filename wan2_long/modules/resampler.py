import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- 基础块 ---------
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner, bias=False),
            nn.GELU(),
            nn.Linear(inner, dim, bias=False),
        )
    def forward(self, x): return self.net(x)

class PerceiverAttention(nn.Module):
    """
    Query = learnable latents
    Key/Value = input tokens (时空token)
    """
    def __init__(self, dim, heads=16, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.norm_x = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.to_q   = nn.Linear(dim, inner, bias=False)
        self.to_kv  = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def _reshape_heads(self, x):
        b, n, d = x.shape
        x = x.view(b, n, self.heads, self.dim_head).transpose(1, 2)  # [B,h,N,d]
        return x

    def forward(self, x, q_latents):
        # x: [B, N, D]  (时空tokens)
        # q_latents: [B, Q, D]
        x = self.norm_x(x)
        q = self.norm_q(q_latents)

        q = self._reshape_heads(self.to_q(q))
        k, v = self.to_kv(x).chunk(2, dim=-1)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        # 软max前缩放，fp16更稳
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        attn = (q * scale) @ (k.transpose(-2, -1) * scale)  # [B,h,Q,N]
        attn = attn.float().softmax(dim=-1).type_as(attn)
        out = attn @ v  # [B,h,Q,d]
        out = out.transpose(1, 2).contiguous().view(x.size(0), q_latents.size(1), -1)
        return self.to_out(out)

# --------- 3D Resampler 主体 ---------
class VideoResampler(nn.Module):
    """
    输入: lr_x [B, F, C, H, W]
    输出: ip_tokens [B, Q, output_dim]
    用途: 将多帧(时空)低分辨率特征压缩为固定数量的『图像提示tokens』，供 IP-Adapter 的 ip_hidden_states 使用。
    """
    def __init__(
        self,
        in_channels=48,             # 输入通道 (RGB=3，若传VAE latent则改为 latent C)
        patch_t=4, patch_h=8, patch_w=8,  # 3D卷积分块
        dim=1024,                  # 内部通道
        depth=4,                   # Perceiver层数
        heads=16, dim_head=64,     # 注意力头设定
        num_queries=16,            # 输出查询token数 (IP常用 8/16/32)
        output_dim=3072,            # 输出到 IP-Adapter 的维度
        ff_mult=4,
        use_pos_emb=True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.use_pos = use_pos_emb

        # 3D Patch Embedding: [B,F,C,H,W] -> [B, D, Ft, Ht, Wt]
        self.patch_embed = nn.Conv3d(
            in_channels, dim,
            kernel_size=(patch_t, patch_h, patch_w),
            stride=(patch_t, patch_h, patch_w),
            padding=0, bias=False
        )

        # 可分解的时空位置编码（时间/高度/宽度）
        if self.use_pos:
            self.temb = nn.Parameter(torch.zeros(1, 1, dim))   # 时间
            self.yemb = nn.Parameter(torch.zeros(1, 1, dim))   # 高
            self.xemb = nn.Parameter(torch.zeros(1, 1, dim))   # 宽
            nn.init.normal_(self.temb, std=0.02)
            nn.init.normal_(self.yemb, std=0.02)
            nn.init.normal_(self.xemb, std=0.02)

        # 可学习查询
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / math.sqrt(dim))

        # Perceiver 堆叠
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, heads=heads, dim_head=dim_head),
                FeedForward(dim=dim, mult=ff_mult),
            ]))

        # 输出投影
        self.proj_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_dim, bias=False),
            nn.LayerNorm(output_dim)
        )

    def _add_pos_emb(self, x_5d):
        # x_5d: [B, D, Ft, Ht, Wt]
        B, D, Ft, Ht, Wt = x_5d.shape
        # 展平成序列前构造可分解PE：t/y/x 单独映射到 D 维并相加
        # 这里用最简单的可学习基向量广播（也可换成正弦PE或小MLP生成）
        t = self.temb.expand(B, Ft, -1)     # [B,Ft,D]
        y = self.yemb.expand(B, Ht, -1)     # [B,Ht,D]
        z = self.xemb.expand(B, Wt, -1)     # [B,Wt,D]

        # 叠加：pe[b, f, y, x, d] = t[b,f,d] + y[b,y,d] + z[b,x,d]
        pe = (
            t[:, :, None, None, :] + 
            y[:, None, :, None, :] + 
            z[:, None, None, :, :]
        )  # [B, Ft, Ht, Wt, D]
        pe = pe.permute(0, 4, 1, 2, 3).contiguous()  # [B, D, Ft, Ht, Wt]
        return x_5d + pe

    def forward(self, lr_x):
        """
        lr_x: [B, F, C, H, W]  注意: 与常见 BCHW 不同, 这里 F 在 C 前
        """
        assert lr_x.dim() == 5, "lr_x must be [B, F, C, H, W]"
        B, F, C, H, W = lr_x.shape
        # 交换到 [B, C, F, H, W] 以喂 Conv3d
        x = lr_x.permute(0, 2, 1, 3, 4).contiguous()

        # 3D patch embed
        x = self.patch_embed(x)  # [B, D, Ft, Ht, Wt]
        if self.use_pos:
            x = self._add_pos_emb(x)

        # 展平为序列 tokens
        B, D, Ft, Ht, Wt = x.shape
        x = x.view(B, D, Ft * Ht * Wt).transpose(1, 2).contiguous()  # [B, N, D], N=Ft*Ht*Wt

        # 查询初始化
        q = self.latents.expand(B, self.num_queries, -1)  # [B, Q, D]

        # Perceiver 堆叠
        for attn, ff in self.layers:
            q = q + attn(x, q)
            q = q + ff(q)

        # 输出到 IP-Adapter 所需维度
        ip_tokens = self.proj_out(q)  # [B, Q, output_dim]
        return ip_tokens
    

if __name__ == "__main__":
    lr_x = torch.randn(1, 3, 48, 30, 52).to("cuda")
    lr_x = lr_x.permute(0, 2, 1, 3, 4).contiguous()
    resampler = VideoResampler().to("cuda")
    ip_tokens = resampler(lr_x)
    print("ip_tokens: ", ip_tokens.shape)