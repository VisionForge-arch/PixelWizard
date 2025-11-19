import os
import json
import random
import argparse
from typing import List, Dict, Any

import cv2


def load_entries(json_path: str) -> List[Dict[str, Any]]:
    """Load json file and return a list of entries."""
    with open(json_path, "r") as f:
        data = json.load(f)
    # 兼容两种结构：直接 list 或者 {"data": [...]}
    if isinstance(data, dict):
        # 自己按需改 key
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        else:
            # 如果是 dict 且不是常见格式，就当成 values
            return list(data.values())
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unsupported JSON format in {json_path}")


def sample_entries(
    entries: List[Dict[str, Any]],
    num_samples: int,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Randomly sample num_samples entries (or all if less)."""
    if len(entries) <= num_samples:
        return entries
    random.seed(seed)
    return random.sample(entries, num_samples)


def extract_first_frame(
    video_path: str,
    save_dir: str,
    prefix: str = ""
) -> str:
    """
    Extract the first frame of a video and save as jpg.
    Returns the saved image path.
    """
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        raise RuntimeError(f"Cannot read first frame from video: {video_path}")

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    if prefix:
        img_name = f"{prefix}_{base_name}.jpg"
    else:
        img_name = f"{base_name}.jpg"

    img_path = os.path.join(save_dir, img_name)
    cv2.imwrite(img_path, frame)

    return img_path


def main():
    parser = argparse.ArgumentParser(
        description="Sample videos for i2v test, extract first frames and create a new JSON."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/UltraVideo/matched_short.json",
        help="Path to the original JSON file containing video paths and captions.",
    )
    parser.add_argument(
        "--output_frames_dir",
        type=str,
        default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/I2V/i2v_test_frames",
        help="Directory to save extracted first frames.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/I2V/i2v_test_samples.json",
        help="Path to save the new JSON with sampled entries and frame paths.",
    )
    parser.add_argument(
        "--video_key",
        type=str,
        default="clip_id",
        help="Key name in JSON entry for video path (e.g., 'video_path', 'video', 'path').",
    )
    parser.add_argument(
        "--caption_key",
        type=str,
        default="brief_description",
        help="Key name in JSON entry for caption.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of videos to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    args = parser.parse_args()

    # 1. 读取 JSON
    print(f"Loading entries from {args.input_json} ...")
    entries = load_entries(args.input_json)
    print(f"Total entries: {len(entries)}")

    # 2. 抽样
    sampled_entries = sample_entries(entries, args.num_samples, args.seed)
    print(f"Sampled entries: {len(sampled_entries)}")

    # 3. 抽首帧并整理新 JSON
    new_entries = []
    os.makedirs(args.output_frames_dir, exist_ok=True)

    for idx, e in enumerate(sampled_entries):
        if args.video_key not in e:
            raise KeyError(
                f"Entry does not contain video key '{args.video_key}': {e.keys()}"
            )
        if args.caption_key not in e:
            raise KeyError(
                f"Entry does not contain caption key '{args.caption_key}': {e.keys()}"
            )

        video_path = e[args.video_key]
        video_path = os.path.join("/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Dataset_fps24", video_path)
        caption = e[args.caption_key]

        # 绝对路径 & 打印一下方便 debug
        print(f"[{idx+1}/{len(sampled_entries)}] video: {video_path}")

        try:
            frame_path = extract_first_frame(
                video_path,
                args.output_frames_dir,
                prefix=f"sample{idx:02d}"
            )
        except Exception as ex:
            print(f"  >> Failed to extract frame from {video_path}: {ex}")
            continue

        # 记录新条目
        new_entries.append(
            {
                "video_path": video_path,
                "caption": caption,
                "frame_path": frame_path,
            }
        )

    # 4. 存新的 JSON
    print(f"Saving new JSON to {args.output_json} ...")
    with open(args.output_json, "w") as f:
        json.dump(new_entries, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()