import argparse
import json
import math
from typing import Any, Dict, Iterable, Optional, Tuple


def to_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return None


def get_nested(item: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur: Any = item
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def iter_records_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter VidGen jsonl by imaging_quality and frames.")
    parser.add_argument("--input", type=str, default="/mnt/vision-gen-ks3/Video_Generation/DataSets/vidgen-1M-sub/jsons/VidGen_1M_video3_recaption.jsonl", help="Input jsonl path.")
    parser.add_argument("--output", type=str, default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/VidGen_1M_video3_recaption.jsonl", help="Output jsonl path (omit for --dry_run).")
    parser.add_argument("--min_imaging_quality", type=float, default=50.0, help="Keep if imaging_quality >= this.")
    parser.add_argument("--frames_min", type=int, default=90, help="Keep if quality.frames >= this.")
    parser.add_argument("--frames_max", type=int, default=160, help="Keep if quality.frames <= this.")
    parser.add_argument("--dry_run", action="store_true", help="Only print stats, do not write output.")
    parser.add_argument("--max_drop_examples", type=int, default=5, help="How many drop examples to print.")
    args = parser.parse_args()

    if not args.dry_run and not args.output:
        raise SystemExit("--output is required unless --dry_run is set")

    total = 0
    kept = 0
    dropped = {
        "missing_imaging_quality": 0,
        "low_imaging_quality": 0,
        "missing_frames": 0,
        "frames_out_of_range": 0,
    }
    drop_examples = []

    out_f = None
    if not args.dry_run:
        out_f = open(args.output, "w")

    try:
        for item in iter_records_jsonl(args.input):
            total += 1
            vid = item.get("vid")

            imaging_quality = to_number(item.get("imaging_quality"))
            frames = to_number(get_nested(item, ("quality", "frames")))

            if imaging_quality is None:
                dropped["missing_imaging_quality"] += 1
                if len(drop_examples) < args.max_drop_examples:
                    drop_examples.append({"vid": vid, "reason": "missing_imaging_quality"})
                continue
            if imaging_quality < float(args.min_imaging_quality):
                dropped["low_imaging_quality"] += 1
                if len(drop_examples) < args.max_drop_examples:
                    drop_examples.append({"vid": vid, "reason": "low_imaging_quality", "imaging_quality": imaging_quality})
                continue

            if frames is None:
                dropped["missing_frames"] += 1
                if len(drop_examples) < args.max_drop_examples:
                    drop_examples.append({"vid": vid, "reason": "missing_frames"})
                continue

            if not (args.frames_min <= frames <= args.frames_max):
                dropped["frames_out_of_range"] += 1
                if len(drop_examples) < args.max_drop_examples:
                    drop_examples.append({"vid": vid, "reason": "frames_out_of_range", "frames": frames})
                continue

            kept += 1
            if out_f is not None:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
    finally:
        if out_f is not None:
            out_f.close()

    stats = {
        "total": total,
        "kept": kept,
        "dropped": dropped,
        "kept_ratio": (kept / total) if total else 0.0,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if drop_examples:
        print("drop_examples:", json.dumps(drop_examples, ensure_ascii=False))


if __name__ == "__main__":
    main()
