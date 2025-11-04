import torch
import matplotlib.pyplot as plt
import os


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
    freqs = torch.outer(torch.arange(max_seq_len), base)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs, base


if __name__ == "__main__":
    out_dir = "/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/rope_viz"
    os.makedirs(out_dir, exist_ok=True)

    max_seq_len = 1024
    dim = 128

    freqs_orig, base_orig = rope_params(max_seq_len, dim, scaling="none")
    freqs_yarn, base_yarn = rope_params(max_seq_len, dim, scaling="yarn", factor=8.0, yarn_alpha=0.8)

    # === 图1：频率基底对比 ===
    plt.figure(figsize=(8,4))
    plt.plot(base_orig.numpy(), label="Original RoPE", lw=2)
    plt.plot(base_yarn.numpy(), label="YARN-scaled RoPE", lw=2)
    plt.xlabel("Dimension index")
    plt.ylabel("Base frequency (θ^{-2i/d})")
    plt.title("Frequency scaling effect (Original vs YARN)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rope_base_comparison.png"), dpi=300)

    # === 图2：相位增长可视化 ===
    pos = torch.arange(max_seq_len)
    low, high = 0, dim // 4
    ang_orig_low = pos * base_orig[low]
    ang_orig_high = pos * base_orig[high]
    ang_yarn_low = pos * base_yarn[low]
    ang_yarn_high = pos * base_yarn[high]

    plt.figure(figsize=(8,4))
    plt.plot(pos, ang_orig_high, label=f"Orig high-freq (dim={high})", color='tab:blue')
    plt.plot(pos, ang_yarn_high, label=f"YARN high-freq (dim={high})", color='tab:orange')
    plt.plot(pos, ang_orig_low, '--', label=f"Orig low-freq (dim={low})", color='tab:blue', alpha=0.6)
    plt.plot(pos, ang_yarn_low, '--', label=f"YARN low-freq (dim={low})", color='tab:orange', alpha=0.6)
    plt.xlabel("Position index")
    plt.ylabel("Phase angle (radians)")
    plt.title("Phase growth per position (Original vs YARN)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rope_phase_growth.png"), dpi=300)

    # === 图3：复平面旋转轨迹 ===
    plt.figure(figsize=(5,5))
    pos = torch.arange(max_seq_len)
    freq_idx = dim // 4
    orig_traj = torch.exp(1j * (pos * base_orig[freq_idx]))
    yarn_traj = torch.exp(1j * (pos * base_yarn[freq_idx]))

    plt.plot(orig_traj.real.numpy(), orig_traj.imag.numpy(),
             label="Original RoPE trajectory", color='tab:blue')
    plt.plot(yarn_traj.real.numpy(), yarn_traj.imag.numpy(),
             label="YARN trajectory", color='tab:orange')
    plt.xlabel("cos(angle)")
    plt.ylabel("sin(angle)")
    plt.title(f"Complex plane rotation (dim={freq_idx})")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rope_complex_rotation.png"), dpi=300)

    print(f"✅ 可视化已保存到: {os.path.abspath(out_dir)}")