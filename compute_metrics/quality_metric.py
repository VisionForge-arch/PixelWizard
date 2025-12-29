#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute HD-MSE / HD-LPIPS on generated videos, and optionally HD-FVD (needs reference set).

Video loading: torchvision.io.read_video (supports mp4, etc.)
HD-MSE/HD-LPIPS: for k in {3,4,5}, downsample by factor 2^k then upsample back.

HD-FVD: patchify HR frames into Hl x Wl patches, extract I3D features on each patch-video,
        then compute Fréchet distance between gen-patch features and ref-patch features.

Dependencies:
  pip install torch torchvision lpips scipy tqdm
  (Optional for better video IO: pip install av)

Usage:
  python hd_metrics.py --gen_dir /path/to/gen_videos
  python hd_metrics.py --gen_dir /path/to/gen_videos --ref_dir /path/to/ref_videos --compute_fvd
"""

import os
import glob
import argparse
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision
from torchvision.io import read_video

import lpips

try:
    from scipy import linalg
except ImportError:
    linalg = None


VIDEO_EXTS = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm")


def list_videos(folder: str) -> List[str]:
    paths = []
    for ext in VIDEO_EXTS:
        paths += glob.glob(os.path.join(folder, ext))
    paths = sorted(paths)
    return paths


def load_video_tchw(path: str, max_frames: Optional[int] = None) -> torch.Tensor:
    """
    Returns: video in [T, C, H, W], float32 in [0,1]
    """
    v, _, _ = read_video(path, pts_unit="sec")  # [T, H, W, C], uint8
    if v.numel() == 0:
        raise RuntimeError(f"Empty video: {path}")
    v = v.to(torch.float32) / 255.0
    v = v.permute(0, 3, 1, 2).contiguous()  # [T,C,H,W]
    if max_frames is not None and v.shape[0] > max_frames:
        # uniform sampling
        idx = torch.linspace(0, v.shape[0] - 1, steps=max_frames).round().long()
        v = v[idx]
    return v


@torch.no_grad()
def hd_mse_single(video_tchw: torch.Tensor, ks=(3, 4, 5), down_mode="bilinear", up_mode="bilinear") -> float:
    """
    video_tchw: [T,C,H,W] in [0,1]
    Return: sum_k MSE(v, up(down(v, 2^k)))
    """
    T, C, H, W = video_tchw.shape
    v = video_tchw.unsqueeze(0)  # [1,T,C,H,W]
    # operate per-frame via reshape to [B*T, C, H, W]
    x = v.reshape(-1, C, H, W)

    total = 0.0
    for k in ks:
        s = 2 ** k
        h2 = max(1, H // s)
        w2 = max(1, W // s)
        xd = F.interpolate(x, size=(h2, w2), mode=down_mode, align_corners=False if down_mode in ("bilinear", "bicubic") else None)
        xu = F.interpolate(xd, size=(H, W), mode=up_mode, align_corners=False if up_mode in ("bilinear", "bicubic") else None)
        mse = F.mse_loss(xu, x, reduction="mean").item()
        total += mse
    return float(total)


def load_lpips_alexnet(device="cuda",
                       weight_path="/mnt/nas01-ak/IndividualDirs/wenxueli/Weight/alexnet-owt-7be5be79.pth"):
    loss_fn = lpips.LPIPS(net="alex", model_path=weight_path)

    loss_fn.eval().to(device)
    return loss_fn

@torch.no_grad()
def hd_lpips_single(video_tchw: torch.Tensor, lpips_fn, ks=(3, 4, 5),
                    down_mode="bilinear", up_mode="bilinear") -> float:
    """
    LPIPS on (v, up(down(v))) frame-wise, average over frames, then sum over k.
    lpips expects inputs in [-1,1], shape [N,3,H,W]
    """
    T, C, H, W = video_tchw.shape
    assert C == 3, "LPIPS expects 3-channel RGB."

    x = video_tchw  # [T,3,H,W] in [0,1]
    x_n = x * 2.0 - 1.0

    total = 0.0
    for k in ks:
        s = 2 ** k
        h2 = max(1, H // s)
        w2 = max(1, W // s)
        xd = F.interpolate(x, size=(h2, w2), mode=down_mode, align_corners=False if down_mode in ("bilinear", "bicubic") else None)
        xu = F.interpolate(xd, size=(H, W), mode=up_mode, align_corners=False if up_mode in ("bilinear", "bicubic") else None)
        xu_n = xu * 2.0 - 1.0

        # batch frames
        d = lpips_fn(x_n, xu_n)  # [T,1,1,1] or [T,1]
        total += float(d.mean().item())
    return float(total)


def patchify_video(video_tchw: torch.Tensor, patch_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Split [T,C,H,W] into non-overlapping patches of size (ph,pw).
    Returns: patches [P, T, C, ph, pw]
    """
    T, C, H, W = video_tchw.shape
    ph, pw = patch_hw
    Hc = (H // ph) * ph
    Wc = (W // pw) * pw
    v = video_tchw[:, :, :Hc, :Wc]  # crop
    # [T,C,Hc,Wc] -> [T,C,Hp,ph,Wp,pw] -> [Hp*Wp, T, C, ph, pw]
    Hp = Hc // ph
    Wp = Wc // pw
    v = v.reshape(T, C, Hp, ph, Wp, pw).permute(2, 4, 0, 1, 3, 5).contiguous()
    patches = v.reshape(Hp * Wp, T, C, ph, pw)
    return patches


def uniform_sample_t(video_tchw: torch.Tensor, num_frames: int) -> torch.Tensor:
    T = video_tchw.shape[0]
    if T == num_frames:
        return video_tchw
    if T < num_frames:
        # pad by repeating last frame
        pad = video_tchw[-1:].repeat(num_frames - T, 1, 1, 1)
        return torch.cat([video_tchw, pad], dim=0)
    idx = torch.linspace(0, T - 1, steps=num_frames).round().long()
    return video_tchw[idx]


def load_i3d_feature_extractor(device: str = "cuda"):
    """
    Uses torchvision's video model as a practical I3D-like feature extractor.
    NOTE: If you strictly need "I3D network used by FVD", you may swap this with a true I3D implementation.
    """
    # r3d_18 is available in torchvision; many FVD codebases use I3D(Kinetics).
    # This is a pragmatic, runnable default.
    model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


@torch.no_grad()
def extract_patch_video_features(video_paths: List[str],
                                 device: str,
                                 patch_hw=(224, 224),
                                 num_frames=16,
                                 max_videos: Optional[int] = None,
                                 batch_size: int = 8) -> np.ndarray:
    """
    For each video: crop->patchify->sample frames -> extract features per patch-video.
    Returns all features as [N, D] numpy array.
    """
    model = load_i3d_feature_extractor(device=device)

    feats = []
    use_paths = video_paths[:max_videos] if max_videos is not None else video_paths

    for p in tqdm(use_paths, desc="Extracting patch features"):
        v = load_video_tchw(p)                       # [T,C,H,W]
        v = uniform_sample_t(v, num_frames)          # [Tf,C,H,W]
        patches = patchify_video(v, patch_hw)        # [P,Tf,C,ph,pw]
        if patches.shape[0] == 0:
            continue

        # model expects [B,C,T,H,W]
        P, Tf, C, ph, pw = patches.shape
        x = patches.permute(0, 2, 1, 3, 4).contiguous()  # [P,C,T,H,W]

        # mini-batch over patches
        for i in range(0, P, batch_size):
            xb = x[i:i + batch_size].to(device=device, dtype=torch.float32)
            fb = model(xb)  # [B,D]
            feats.append(fb.detach().cpu().numpy())

    if len(feats) == 0:
        return np.zeros((0, 512), dtype=np.float32)
    return np.concatenate(feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    """
    Standard Fréchet distance between two Gaussians.
    """
    if linalg is None:
        raise ImportError("scipy is required for HD-FVD: pip install scipy")

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def compute_stats(feats: np.ndarray):
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/eval_100/240p_5s/decoded_video")
    parser.add_argument("--ref_dir", type=str, default=None, help="Needed for HD-FVD")
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap when loading for HD-MSE/LPIPS")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ks", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--compute_lpips", action="store_true", default=True)
    parser.add_argument("--compute_fvd", action="store_true", default=False)
    parser.add_argument("--patch_h", type=int, default=224)
    parser.add_argument("--patch_w", type=int, default=224)
    parser.add_argument("--fvd_num_frames", type=int, default=16)
    parser.add_argument("--fvd_batch_size", type=int, default=8)
    args = parser.parse_args()

    gen_paths = list_videos(args.gen_dir)
    if args.max_videos is not None:
        gen_paths = gen_paths[:args.max_videos]
    assert len(gen_paths) > 0, f"No videos found in {args.gen_dir}"

    # LPIPS model
    lpips_fn = None
    if args.compute_lpips:
        if lpips is None:
            raise ImportError("lpips is required: pip install lpips")
        lpips_fn = load_lpips_alexnet(
            device=args.device,
            weight_path="/mnt/nas01-ak/IndividualDirs/wenxueli/Weight/alexnet-owt-7be5be79.pth"
        )

    # ---- HD-MSE / HD-LPIPS on generated videos only ----
    hd_mse_vals = []
    hd_lpips_vals = []

    for p in tqdm(gen_paths, desc="HD-MSE/HD-LPIPS"):
        v = load_video_tchw(p, max_frames=args.max_frames)  # [T,C,H,W] in [0,1]
        hd_mse_vals.append(hd_mse_single(v, ks=tuple(args.ks)))
        if lpips_fn is not None:
            hd_lpips_vals.append(hd_lpips_single(v, lpips_fn, ks=tuple(args.ks)))

    print("======== Results (Generated videos) ========")
    print(f"Num videos: {len(gen_paths)}")
    print(f"HD-MSE  (sum over k={args.ks}): mean={np.mean(hd_mse_vals):.6f}  std={np.std(hd_mse_vals):.6f}")
    if lpips_fn is not None:
        print(f"HD-LPIPS(sum over k={args.ks}): mean={np.mean(hd_lpips_vals):.6f}  std={np.std(hd_lpips_vals):.6f}")

    # ---- Optional HD-FVD (needs ref set) ----
    if args.compute_fvd:
        assert args.ref_dir is not None, "--compute_fvd requires --ref_dir"
        ref_paths = list_videos(args.ref_dir)
        if args.max_videos is not None:
            ref_paths = ref_paths[:args.max_videos]
        assert len(ref_paths) > 0, f"No videos found in {args.ref_dir}"

        patch_hw = (args.patch_h, args.patch_w)
        gen_feats = extract_patch_video_features(
            gen_paths, device=args.device, patch_hw=patch_hw,
            num_frames=args.fvd_num_frames, max_videos=args.max_videos,
            batch_size=args.fvd_batch_size
        )
        ref_feats = extract_patch_video_features(
            ref_paths, device=args.device, patch_hw=patch_hw,
            num_frames=args.fvd_num_frames, max_videos=args.max_videos,
            batch_size=args.fvd_batch_size
        )

        mu1, s1 = compute_stats(gen_feats)
        mu2, s2 = compute_stats(ref_feats)
        hd_fvd = frechet_distance(mu1, s1, mu2, s2)

        print("======== HD-FVD (Gen vs Ref) ========")
        print(f"Patches(gen): {gen_feats.shape[0]}   Patches(ref): {ref_feats.shape[0]}")
        print(f"HD-FVD: {hd_fvd:.4f}")
    else:
        if args.ref_dir is not None:
            print("Note: ref_dir provided but --compute_fvd not set, skipping HD-FVD.")


if __name__ == "__main__":
    main()