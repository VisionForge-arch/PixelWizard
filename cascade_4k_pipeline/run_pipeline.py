#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_DIR = REPO_ROOT / "Wan2.2"
STAGE2_DIR = REPO_ROOT / "Wan2.2_Cross"
STAGE1_SCRIPT = "generate_multiple.py"
STAGE2_SCRIPT = "generate_multiple_upsample_shortcut.py"


def parse_optional_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def bool_to_cli(value: bool | None) -> str | None:
    if value is None:
        return None
    return "True" if value else "False"


def add_value_arg(cmd: list[str], name: str, value: object | None) -> None:
    if value is None:
        return
    cmd.extend([name, str(value)])


def add_flag_arg(cmd: list[str], name: str, enabled: bool) -> None:
    if enabled:
        cmd.append(name)


def ensure_file(path: Path | None, label: str) -> None:
    if path is None:
        raise FileNotFoundError(f"{label} is required")
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def ensure_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def build_runner_prefix(nproc: int, python_bin: str, torchrun_bin: str, script_name: str) -> list[str]:
    ensure_positive_int("nproc", nproc)
    if nproc == 1:
        return [python_bin, script_name]
    return [torchrun_bin, "--standalone", f"--nproc_per_node={nproc}", script_name]


def build_stage1_command(args, prompt_file: Path, output_dir: Path, manifest_path: Path) -> list[str]:
    cmd = build_runner_prefix(
        nproc=args.stage1_nproc,
        python_bin=args.python_bin,
        torchrun_bin=args.torchrun_bin,
        script_name=STAGE1_SCRIPT,
    )
    add_value_arg(cmd, "--task", args.stage1_task)
    add_value_arg(cmd, "--size", args.stage1_size)
    add_value_arg(cmd, "--frame_num", args.stage1_frame_num)
    add_value_arg(cmd, "--sample_solver", args.stage1_sample_solver)
    add_value_arg(cmd, "--sample_steps", args.stage1_sample_steps)
    add_value_arg(cmd, "--sample_shift", args.stage1_sample_shift)
    add_value_arg(cmd, "--base_seed", args.stage1_base_seed)
    add_value_arg(cmd, "--prompt_file", prompt_file)
    add_value_arg(cmd, "--save_file", output_dir)
    add_value_arg(cmd, "--manifest_file", manifest_path)
    add_value_arg(cmd, "--ckpt_dir", args.stage1_ckpt_dir)
    add_value_arg(cmd, "--wan_ckpt", args.stage1_wan_ckpt)
    add_value_arg(cmd, "--offload_model", bool_to_cli(args.stage1_offload_model))
    add_value_arg(cmd, "--save_latent", bool_to_cli(args.stage1_save_latent))
    add_value_arg(cmd, "--save_video", bool_to_cli(args.stage1_save_video))
    add_value_arg(cmd, "--ulysses_size", args.stage1_ulysses_size)
    add_flag_arg(cmd, "--t5_fsdp", args.stage1_t5_fsdp)
    add_flag_arg(cmd, "--t5_cpu", args.stage1_t5_cpu)
    add_flag_arg(cmd, "--dit_fsdp", args.stage1_dit_fsdp)
    add_flag_arg(cmd, "--use_ema", args.stage1_use_ema)
    add_flag_arg(cmd, "--use_lora", args.stage1_use_lora)
    return cmd


def build_stage2_command(args, manifest_path: Path, output_dir: Path) -> list[str]:
    cmd = build_runner_prefix(
        nproc=args.stage2_nproc,
        python_bin=args.python_bin,
        torchrun_bin=args.torchrun_bin,
        script_name=STAGE2_SCRIPT,
    )
    add_value_arg(cmd, "--task", args.stage2_task)
    add_value_arg(cmd, "--size", args.stage2_size)
    add_value_arg(cmd, "--frame_num", args.stage2_frame_num)
    add_value_arg(cmd, "--sample_solver", args.stage2_sample_solver)
    add_value_arg(cmd, "--sample_steps", args.stage2_sample_steps)
    add_value_arg(cmd, "--sample_shift", args.stage2_sample_shift)
    add_value_arg(cmd, "--base_seed", args.stage2_base_seed)
    add_value_arg(cmd, "--prompt_file", manifest_path)
    add_value_arg(cmd, "--save_file", output_dir)
    add_value_arg(cmd, "--ckpt_dir", args.stage2_ckpt_dir)
    add_value_arg(cmd, "--wan_ckpt", args.stage2_wan_ckpt)
    add_value_arg(cmd, "--offload_model", bool_to_cli(args.stage2_offload_model))
    add_value_arg(cmd, "--ulysses_size", args.stage2_ulysses_size)
    add_flag_arg(cmd, "--t5_fsdp", args.stage2_t5_fsdp)
    add_flag_arg(cmd, "--t5_cpu", args.stage2_t5_cpu)
    add_flag_arg(cmd, "--dit_fsdp", args.stage2_dit_fsdp)
    add_flag_arg(cmd, "--use_ema", args.stage2_use_ema)
    return cmd


def run_command(cmd: list[str], cwd: Path, devices: str | None, dry_run: bool) -> None:
    env = os.environ.copy()
    if devices:
        env["CUDA_VISIBLE_DEVICES"] = devices
    print(f"[run] cwd={cwd}")
    print(f"[run] cmd={shlex.join(cmd)}")
    if devices:
        print(f"[run] CUDA_VISIBLE_DEVICES={devices}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def resolve_paths(args):
    output_root = Path(args.output_root).expanduser().resolve()
    stage1_output_dir = Path(args.stage1_output_dir).expanduser().resolve() if args.stage1_output_dir else output_root / "stage1_latents"
    stage2_output_dir = Path(args.stage2_output_dir).expanduser().resolve() if args.stage2_output_dir else output_root / "stage2_latents"
    manifest_path = Path(args.manifest_path).expanduser().resolve() if args.manifest_path else output_root / "manifests" / "stage1_to_stage2.json"
    prompt_file = Path(args.prompt_file).expanduser().resolve() if args.prompt_file else None
    input_manifest = Path(args.stage1_manifest).expanduser().resolve() if args.stage1_manifest else manifest_path
    return output_root, stage1_output_dir, stage2_output_dir, manifest_path, prompt_file, input_manifest


def validate_args(args, prompt_file: Path | None, input_manifest: Path) -> None:
    if not STAGE1_DIR.is_dir():
        raise FileNotFoundError(f"Missing stage-1 directory: {STAGE1_DIR}")
    if not STAGE2_DIR.is_dir():
        raise FileNotFoundError(f"Missing stage-2 directory: {STAGE2_DIR}")

    ensure_positive_int("stage1_nproc", args.stage1_nproc)
    ensure_positive_int("stage2_nproc", args.stage2_nproc)
    ensure_positive_int("stage1_ulysses_size", args.stage1_ulysses_size)
    ensure_positive_int("stage2_ulysses_size", args.stage2_ulysses_size)

    if args.mode in {"full", "stage1"}:
        ensure_file(prompt_file, "prompt file")
    if args.mode == "stage2":
        ensure_file(input_manifest, "stage-1 manifest")

    if args.stage1_ulysses_size > 1 and args.stage1_ulysses_size != args.stage1_nproc:
        raise ValueError("stage1_ulysses_size must match stage1_nproc when ulysses_size > 1")
    if args.stage2_ulysses_size > 1 and args.stage2_ulysses_size != args.stage2_nproc:
        raise ValueError("stage2_ulysses_size must match stage2_nproc when ulysses_size > 1")
    if not args.stage1_save_latent and not args.stage1_save_video:
        raise ValueError("At least one of stage1_save_latent or stage1_save_video must be enabled.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified two-stage 4K cascade pipeline for Wan2.2 -> Wan2.2_Cross."
    )
    parser.add_argument(
        "--mode",
        choices=("full", "stage1", "stage2"),
        default="full",
        help="Run the whole cascade, only stage-1, or only stage-2.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory for pipeline outputs. Defaults to stage1_latents, stage2_latents, and manifests under this folder.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Prompt text file for stage-1. Required for full/stage1 mode.",
    )
    parser.add_argument(
        "--stage1-manifest",
        default=None,
        help="Existing stage-1 manifest JSON. Required for stage2-only mode when manifest_path does not already exist.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Manifest path written by stage-1 and consumed by stage-2. Default: <output-root>/manifests/stage1_to_stage2.json",
    )
    parser.add_argument(
        "--stage1-output-dir",
        default=None,
        help="Override stage-1 latent output directory. Default: <output-root>/stage1_latents",
    )
    parser.add_argument(
        "--stage2-output-dir",
        default=None,
        help="Override stage-2 latent output directory. Default: <output-root>/stage2_latents",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used when nproc=1.",
    )
    parser.add_argument(
        "--torchrun-bin",
        default="torchrun",
        help="torchrun executable used when nproc>1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the stage commands without executing them.",
    )

    stage1 = parser.add_argument_group("stage1")
    stage1.add_argument("--stage1-devices", default=None, help="CUDA_VISIBLE_DEVICES for stage-1.")
    stage1.add_argument("--stage1-nproc", type=int, default=1, help="Number of processes for stage-1.")
    stage1.add_argument("--stage1-task", default="ti2v-5B", help="Stage-1 task name.")
    stage1.add_argument("--stage1-size", default="448*256", help="Stage-1 output size.")
    stage1.add_argument("--stage1-frame-num", type=int, default=121, help="Stage-1 frame count.")
    stage1.add_argument("--stage1-sample-solver", default="unipc", choices=("unipc", "dpm++"), help="Stage-1 sampler.")
    stage1.add_argument("--stage1-sample-steps", type=int, default=50, help="Stage-1 sampling steps.")
    stage1.add_argument("--stage1-sample-shift", type=float, default=None, help="Stage-1 sampling shift.")
    stage1.add_argument("--stage1-base-seed", type=int, default=0, help="Stage-1 base seed.")
    stage1.add_argument("--stage1-ckpt-dir", default=None, help="Stage-1 checkpoint directory.")
    stage1.add_argument("--stage1-wan-ckpt", default=None, help="Stage-1 model checkpoint.")
    stage1.add_argument("--stage1-offload-model", type=parse_optional_bool, default=None, help="Whether stage-1 offloads the model to CPU. Omit to keep the script default.")
    stage1.add_argument("--stage1-save-latent", type=parse_optional_bool, default=True, help="Whether to save stage-1 latent .pt files.")
    stage1.add_argument("--stage1-save-video", type=parse_optional_bool, default=False, help="Whether to decode and save stage-1 videos as .mp4 files.")
    stage1.add_argument("--stage1-ulysses-size", type=int, default=1, help="Stage-1 ulysses size.")
    stage1.add_argument("--stage1-t5-fsdp", action="store_true", help="Enable stage-1 T5 FSDP.")
    stage1.add_argument("--stage1-t5-cpu", action="store_true", help="Place stage-1 T5 on CPU.")
    stage1.add_argument("--stage1-dit-fsdp", action="store_true", help="Enable stage-1 DiT FSDP.")
    stage1.add_argument("--stage1-use-ema", action="store_true", help="Use EMA weights in stage-1.")
    stage1.add_argument("--stage1-use-lora", action="store_true", help="Use LoRA weights in stage-1.")

    stage2 = parser.add_argument_group("stage2")
    stage2.add_argument("--stage2-devices", default=None, help="CUDA_VISIBLE_DEVICES for stage-2.")
    stage2.add_argument("--stage2-nproc", type=int, default=1, help="Number of processes for stage-2.")
    stage2.add_argument("--stage2-task", default="ti2v-5B", help="Stage-2 task name.")
    stage2.add_argument("--stage2-size", default="3840*2144", help="Stage-2 output size.")
    stage2.add_argument("--stage2-frame-num", type=int, default=121, help="Stage-2 frame count.")
    stage2.add_argument("--stage2-sample-solver", default="unipc", choices=("unipc", "dpm++"), help="Stage-2 sampler.")
    stage2.add_argument("--stage2-sample-steps", type=int, default=5, help="Stage-2 sampling steps.")
    stage2.add_argument("--stage2-sample-shift", type=float, default=5.8, help="Stage-2 sampling shift.")
    stage2.add_argument("--stage2-base-seed", type=int, default=1, help="Stage-2 base seed.")
    stage2.add_argument("--stage2-ckpt-dir", default=None, help="Stage-2 checkpoint directory.")
    stage2.add_argument("--stage2-wan-ckpt", default=None, help="Stage-2 model checkpoint.")
    stage2.add_argument("--stage2-offload-model", type=parse_optional_bool, default=None, help="Whether stage-2 offloads the model to CPU. Omit to keep the script default.")
    stage2.add_argument("--stage2-ulysses-size", type=int, default=1, help="Stage-2 ulysses size.")
    stage2.add_argument("--stage2-t5-fsdp", action="store_true", help="Enable stage-2 T5 FSDP.")
    stage2.add_argument("--stage2-t5-cpu", action="store_true", help="Place stage-2 T5 on CPU.")
    stage2.add_argument("--stage2-dit-fsdp", action="store_true", help="Enable stage-2 DiT FSDP.")
    stage2.add_argument("--stage2-use-ema", action="store_true", help="Use EMA weights in stage-2.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root, stage1_output_dir, stage2_output_dir, manifest_path, prompt_file, input_manifest = resolve_paths(args)
    validate_args(args, prompt_file, input_manifest)

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        stage1_output_dir.mkdir(parents=True, exist_ok=True)
        stage2_output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode in {"full", "stage1"}:
        stage1_cmd = build_stage1_command(args, prompt_file, stage1_output_dir, manifest_path)
        run_command(stage1_cmd, cwd=STAGE1_DIR, devices=args.stage1_devices, dry_run=args.dry_run)

    if args.mode in {"full", "stage2"}:
        stage2_manifest = manifest_path if args.mode == "full" else input_manifest
        if not args.dry_run:
            ensure_file(stage2_manifest, "stage-1 manifest")
        stage2_cmd = build_stage2_command(args, stage2_manifest, stage2_output_dir)
        run_command(stage2_cmd, cwd=STAGE2_DIR, devices=args.stage2_devices, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
