import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# ========== 你的 rope_params（含 YARN） ==========
def rope_params(max_seq_len, dim, theta=10000,
                scaling: str = "none",   # "none" | "yarn"
                factor: float = 8.0,
                yarn_alpha: float = 0.8,
                yarn_short_factor: float = 1.0):
    assert dim % 2 == 0
    idx = torch.arange(0, dim, 2).to(torch.float64)
    base = 1.0 / torch.pow(theta, idx.div(dim))
    if scaling == "yarn" and factor != 1.0:
        t = torch.linspace(0.0, 1.0, base.numel(), dtype=torch.float64)
        half = base.numel() // 2
        yarn_scale = torch.ones_like(base)
        if half > 0:
            yarn_scale[:half] = (factor ** (t[:half] ** yarn_alpha))**(-1) * yarn_short_factor
        yarn_scale[half:] = (factor ** (t[half:] ** yarn_alpha))**(-1)
        base = base * yarn_scale
    freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float64), base)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs, base

if __name__ == "__main__":
    out_dir = "/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/rope_viz"
    os.makedirs(out_dir, exist_ok=True)

    max_seq_len = 1024
    dim = 128
    factor = 8.0
    yarn_alpha = 0.8

    freqs_o, base_o = rope_params(max_seq_len, dim, scaling="none")
    freqs_y, base_y = rope_params(max_seq_len, dim, scaling="yarn", factor=factor, yarn_alpha=yarn_alpha)

    # ---------- 图1：频率基底曲线 ----------
    plt.figure(figsize=(8,4))
    plt.plot(base_o.numpy(), label="Original", lw=2)
    plt.plot(base_y.numpy(), label=f"YARN (factor={factor}, alpha={yarn_alpha})", lw=2)
    plt.xlabel("Dimension index (k)")
    plt.ylabel("Base frequency (θ^{-2k/d})")
    plt.title("Base frequency vs. dimension (Original vs. YARN)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "freq_base_comparison.png"), dpi=300); plt.close()

    # ---------- 图2：相位斜率热力图（就是 base，最直观） ----------
    # 把 base 重复到 [dim/2, 1] 形成“热力图”的列
    slope_mat = torch.stack([base_o, base_y], dim=1).numpy()  # [dim/2, 2]
    plt.figure(figsize=(6,6))
    plt.imshow(np.log10(slope_mat + 1e-30), aspect='auto', origin='lower', cmap='viridis')
    plt.yticks([0, base_o.numel()//2, base_o.numel()-1], ["low-freq","mid","high-freq"])
    plt.xticks([0,1], ["Original","YARN"])
    plt.colorbar(label="log10(phase slope = base)")
    plt.title("Phase slope heatmap across dimensions")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase_slope_heatmap.png"), dpi=300); plt.close()

    # ---------- 图3：相位 wrapped（mod 2π） ----------
    pos = torch.arange(max_seq_len, dtype=torch.float64)
    # 选取多条频率曲线画在一起（低/中/高）
    idxs = [0, dim//16, dim//8, dim//4, dim//2 - 1]
    plt.figure(figsize=(9,5))
    for k in idxs:
        ang_o = (pos * base_o[k]) % (2*np.pi)
        ang_y = (pos * base_y[k]) % (2*np.pi)
        plt.plot(pos, ang_o, alpha=0.7, lw=1.5, label=f"Orig k={k}")
        plt.plot(pos, ang_y, alpha=0.7, lw=1.5, linestyle="--", label=f"YARN k={k}")
    plt.xlabel("Position index")
    plt.ylabel("Phase mod 2π")
    plt.title("Wrapped phase (mod 2π): Original vs YARN for multiple dims")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase_wrapped_mod2pi.png"), dpi=300); plt.close()

    # ---------- 图4：相位 Unwrapped & 归一化 ----------
    # 把每条曲线除以其最大相位值，叠加对比（去掉尺度差异）
    plt.figure(figsize=(9,5))
    for k in idxs:
        ang_o = pos * base_o[k]
        ang_y = pos * base_y[k]
        ang_o = (ang_o - ang_o.min()) / (ang_o.max() - ang_o.min() + 1e-12)
        ang_y = (ang_y - ang_y.min()) / (ang_y.max() - ang_y.min() + 1e-12)
        plt.plot(pos, ang_o, lw=1.5, label=f"Orig k={k}")
        plt.plot(pos, ang_y, lw=1.5, linestyle="--", label=f"YARN k={k}")
    plt.xlabel("Position index")
    plt.ylabel("Normalized phase [0,1]")
    plt.title("Normalized phase growth (remove absolute scale)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase_unwrapped_normalized.png"), dpi=300); plt.close()

    # ---------- 图5：代表维度的小面板（更清晰） ----------
    reps = [0, dim//8, dim//4, dim//2 - 1]
    fig, axes = plt.subplots(2, 2, figsize=(10,6), sharex=True)
    axes = axes.ravel()
    for ax, k in zip(axes, reps):
        ang_o = pos * base_o[k]
        ang_y = pos * base_y[k]
        ax.plot(pos, ang_o, label="Original")
        ax.plot(pos, ang_y, label="YARN", linestyle="--")
        ax.set_title(f"Phase growth @ dim={k}")
        ax.grid(alpha=0.3)
    axes[0].legend()
    for ax in axes: ax.set_xlabel("Position"); ax.set_ylabel("Phase (rad)")
    fig.suptitle("Phase growth per selected dimensions", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phase_growth_panel.png"), dpi=300); plt.close()

    # ---------- 图6：复平面旋转轨迹面板 ----------
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    k_list = [0, dim//8, dim//4]
    for ax, k in zip(axes, k_list):
        traj_o = torch.exp(1j * (pos * base_o[k]))
        traj_y = torch.exp(1j * (pos * base_y[k]))
        ax.plot(traj_o.real.numpy(), traj_o.imag.numpy(), label="Original")
        ax.plot(traj_y.real.numpy(), traj_y.imag.numpy(), label="YARN", linestyle="--")
        ax.set_title(f"Complex rotation (dim={k})")
        ax.set_xlabel("cos"); ax.set_ylabel("sin"); ax.axis("equal"); ax.grid(alpha=0.3)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "complex_rotation_panel.png"), dpi=300); plt.close()

    print(f"✅ Saved visualizations to: {out_dir}")