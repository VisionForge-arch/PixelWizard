import argparse
import json
import math
from collections import Counter
from statistics import mean, median
from typing import Iterable, List


def load_records(path: str) -> Iterable[dict]:
    with open(path, "r") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        else:
            data = json.load(f)
            if isinstance(data, dict):
                yield data
            else:
                for item in data:
                    yield item


def extract_fps(records: Iterable[dict]) -> List[float]:
    fps_values = []
    for item in records:
        try:
            fps = item["quality"]["fps"]
        except Exception:
            continue
        if isinstance(fps, (int, float)) and not math.isnan(fps):
            fps_values.append(float(fps))
    return fps_values


def describe_fps(fps_values: List[float], round_digits: int, topk: int):
    if not fps_values:
        print("No fps values found.")
        return

    rounded = [round(v, round_digits) for v in fps_values]
    freq = Counter(rounded)
    total = len(fps_values)
    print(f"Total samples: {total}")
    print(f"Min fps: {min(fps_values):.6f}")
    print(f"Max fps: {max(fps_values):.6f}")
    print(f"Mean fps: {mean(fps_values):.6f}")
    print(f"Median fps: {median(fps_values):.6f}")
    print(f"Unique fps (rounded to {round_digits}): {len(freq)}")
    print(f"Top {topk} most common (rounded):")
    for value, cnt in freq.most_common(topk):
        print(f"  {value}: {cnt}")


def main():
    parser = argparse.ArgumentParser(description="Compute fps statistics for VidGen-1M metadata.")
    parser.add_argument(
        "--input",
        type=str,
        default="/mnt/vision-gen-ks3/Video_Generation/DataSets/vidgen-1M-sub/jsons/VidGen_1M_video3_recaption.jsonl",
        help="Path to metadata json/jsonl file.",
    )
    parser.add_argument("--round_digits", type=int, default=3, help="Round fps before counting frequencies.")
    parser.add_argument("--topk", type=int, default=10, help="How many most common fps buckets to show.")
    args = parser.parse_args()

    records = load_records(args.input)
    fps_values = extract_fps(records)
    describe_fps(fps_values, args.round_digits, args.topk)


if __name__ == "__main__":
    main()
