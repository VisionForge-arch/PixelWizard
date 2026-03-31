# Cascade 4K Pipeline

这个目录把当前的两阶段 4K 级联整理成了一个统一入口：

- stage-1: [`Wan2.2/generate_multiple.py`](/Users/wli/Code/UltraVideo/Wan2.2/generate_multiple.py) 生成低分辨率 latent
- stage-2: [`Wan2.2_Cross/generate_multiple_upsample_shortcut.py`](/Users/wli/Code/UltraVideo/Wan2.2_Cross/generate_multiple_upsample_shortcut.py) 读取 stage-1 manifest，做 4K 上采样

## 输出结构

默认会在 `--output-root` 下生成：

- `stage1_latents/`: 第一阶段输出的 `.pt`
- `manifests/stage1_to_stage2.json`: 第二阶段直接可读的 `prompt -> file` 清单
- `stage2_latents/`: 第二阶段输出的 `.pt`

stage-1 的保存行为可以通过开关控制：

- `--stage1-save-latent true|false`: 是否保存第一阶段 latent，默认 `true`
- `--stage1-save-video true|false`: 是否额外把第一阶段结果 decode 成 mp4，默认 `false`

当两者都开启时，manifest 默认优先指向 latent；如果关闭 latent 只保留 video，manifest 会自动指向 mp4，stage-2 会直接读取视频。

## 一键跑完整 pipeline

```bash
python3 /Users/wli/Code/UltraVideo/cascade_4k_pipeline/run_pipeline.py \
  --mode full \
  --output-root /path/to/run_001 \
  --prompt-file /path/to/prompts.txt \
  --stage1-wan-ckpt /path/to/stage1/model.pt \
  --stage2-wan-ckpt /path/to/stage2/model.pt \
  --stage1-save-latent true \
  --stage1-save-video false \
  --stage1-nproc 6 \
  --stage2-nproc 6 \
  --stage1-devices 1,2,3,4,5,6 \
  --stage2-devices 1,2,3,4,5,6 \
  --stage1-dit-fsdp \
  --stage1-t5-fsdp \
  --stage1-ulysses-size 6 \
  --stage2-dit-fsdp \
  --stage2-t5-fsdp \
  --stage2-ulysses-size 6
```

## 只跑第二阶段

如果你已经有 stage-1 manifest，可以直接续跑：

```bash
python3 /Users/wli/Code/UltraVideo/cascade_4k_pipeline/run_pipeline.py \
  --mode stage2 \
  --output-root /path/to/run_001 \
  --stage1-manifest /path/to/run_001/manifests/stage1_to_stage2.json \
  --stage2-wan-ckpt /path/to/stage2/model.pt
```

## 说明

- stage-1 现在支持 `--manifest_file`，会在生成每个 latent 后持续刷新 manifest。
- manifest 中除了 `prompt` 和 `file` 之外，还会带上 `latent_file`、`video_file`、`input_type`、`seed`、`size`、`frame_num` 等字段，stage-2 会自动忽略额外字段。
- `--dry-run` 可以先打印实际会执行的 `python` 或 `torchrun` 命令。
