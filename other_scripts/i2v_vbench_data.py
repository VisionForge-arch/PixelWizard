#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def main():
    parser = argparse.ArgumentParser(
        description="Convert a folder of prompt-named images to caption/frame JSON."
    )
    parser.add_argument("image_dir", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Dataset", help="Folder that holds the images")
    parser.add_argument("output_json", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/I2V/i2v_vbench.json", help="Path to write the JSON list")
    parser.add_argument(
        "--frame-prefix",
        help="Optional override for frame_path (will append the filename to this prefix)",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir).expanduser()
    data = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        caption = path.stem
        frame_path = (
            str(Path(args.frame_prefix) / path.name)
            if args.frame_prefix
            else str(path.resolve())
        )
        data.append({"caption": caption, "frame_path": frame_path})

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
