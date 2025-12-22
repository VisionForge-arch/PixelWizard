import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


def iter_records(path: str) -> Tuple[Iterable[dict], Dict[str, int]]:
    errors = {"empty_line": 0, "json_decode_error": 0, "non_dict_record": 0}

    def _iter() -> Iterable[dict]:
        with open(path, "r") as f:
            if path.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if not line:
                        errors["empty_line"] += 1
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        errors["json_decode_error"] += 1
                        continue
                    if isinstance(item, dict):
                        yield item
                    else:
                        errors["non_dict_record"] += 1
            else:
                data = json.load(f)
                if isinstance(data, dict):
                    yield data
                else:
                    for item in data:
                        if isinstance(item, dict):
                            yield item
                        else:
                            errors["non_dict_record"] += 1

    return _iter(), errors


def get_nested(item: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur: Any = item
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def to_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return None

@dataclass
class StreamingStats:
    name: str
    sample_size: int = 100_000
    _n: int = 0
    _mean: float = 0.0
    _m2: float = 0.0
    _min: float = float("inf")
    _max: float = float("-inf")
    _sample: List[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._m2 += delta * delta2
        self._min = min(self._min, value)
        self._max = max(self._max, value)
        self._reservoir_add(value)

    def _reservoir_add(self, value: float) -> None:
        if self.sample_size <= 0:
            return
        if len(self._sample) < self.sample_size:
            self._sample.append(value)
            return
        j = random.randrange(0, self._n)
        if j < self.sample_size:
            self._sample[j] = value

    def summary(self, quantiles: List[float]) -> Dict[str, Any]:
        if self._n == 0:
            return {"count": 0}
        variance = self._m2 / (self._n - 1) if self._n > 1 else 0.0
        out: Dict[str, Any] = {
            "count": self._n,
            "min": self._min,
            "max": self._max,
            "mean": self._mean,
            "std": math.sqrt(variance),
        }
        if self._sample:
            xs = sorted(self._sample)
            qs = {}
            for q in quantiles:
                if not (0.0 <= q <= 1.0):
                    continue
                idx = int(round(q * (len(xs) - 1)))
                qs[str(q)] = xs[idx]
            out["quantiles"] = qs
            out["quantile_sample_n"] = len(xs)
        return out


def _format_topk(counter: Counter, topk: int) -> List[Dict[str, Any]]:
    return [{"value": k, "count": v} for k, v in counter.most_common(topk)]


def compute_stats(
    records: Iterable[dict],
    round_digits: int,
    topk: int,
    sample_size: int,
    seed: int,
    quantiles: List[float],
) -> Dict[str, Any]:
    random.seed(seed)

    total = 0
    missing = Counter()

    fps_stats = StreamingStats("fps", sample_size=sample_size)
    duration_stats = StreamingStats("duration", sample_size=sample_size)
    frames_stats = StreamingStats("frames", sample_size=sample_size)
    width_stats = StreamingStats("width", sample_size=sample_size)
    height_stats = StreamingStats("height", sample_size=sample_size)
    imaging_quality_stats = StreamingStats("imaging_quality", sample_size=sample_size)
    aspect_ratio_stats = StreamingStats("aspect_ratio", sample_size=sample_size)
    area_stats = StreamingStats("area", sample_size=sample_size)
    caption_len_stats = StreamingStats("caption_len", sample_size=sample_size)
    caption_words_stats = StreamingStats("caption_words", sample_size=sample_size)
    fps_from_frames_stats = StreamingStats("fps_from_frames", sample_size=sample_size)

    fps_rounded = Counter()
    resolution = Counter()
    video_dir = Counter()

    fps_mismatch = 0
    fps_mismatch_examples: List[Dict[str, Any]] = []

    for item in records:
        total += 1
        vid = item.get("vid")

        caption = item.get("caption")
        if isinstance(caption, str):
            caption_len_stats.add(float(len(caption)))
            caption_words_stats.add(float(len(caption.split())))
        else:
            missing["caption"] += 1

        vp = item.get("video_path")
        if isinstance(vp, str):
            # keep it stable: last 3 folders + filename can be noisy; use parent folder
            parts = [p for p in vp.split("/") if p]
            if len(parts) >= 2:
                video_dir["/".join(parts[-2:-1])] += 1
        else:
            missing["video_path"] += 1

        width = to_number(get_nested(item, ("quality", "width")))
        height = to_number(get_nested(item, ("quality", "height")))
        fps = to_number(get_nested(item, ("quality", "fps")))
        duration = to_number(get_nested(item, ("quality", "duration")))
        frames = to_number(get_nested(item, ("quality", "frames")))
        imaging_quality = to_number(item.get("imaging_quality"))

        if width is None:
            missing["quality.width"] += 1
        else:
            width_stats.add(width)
        if height is None:
            missing["quality.height"] += 1
        else:
            height_stats.add(height)
        if fps is None:
            missing["quality.fps"] += 1
        else:
            fps_stats.add(fps)
            fps_rounded[round(fps, round_digits)] += 1
        if duration is None:
            missing["quality.duration"] += 1
        else:
            duration_stats.add(duration)
        if frames is None:
            missing["quality.frames"] += 1
        else:
            frames_stats.add(frames)

        if width is not None and height is not None and width > 0 and height > 0:
            resolution[f"{int(width)}x{int(height)}"] += 1
            aspect_ratio_stats.add(width / height)
            area_stats.add(width * height)
        else:
            missing["quality.resolution"] += 1

        if imaging_quality is None:
            missing["imaging_quality"] += 1
        else:
            imaging_quality_stats.add(imaging_quality)

        if frames is not None and duration is not None and duration > 0:
            fps_ff = frames / duration
            fps_from_frames_stats.add(fps_ff)
            if fps is not None and abs(fps - fps_ff) > 0.5:
                fps_mismatch += 1
                if len(fps_mismatch_examples) < 5:
                    fps_mismatch_examples.append(
                        {
                            "vid": vid,
                            "fps": fps,
                            "frames": frames,
                            "duration": duration,
                            "fps_from_frames": fps_ff,
                        }
                    )

    return {
        "total_records": total,
        "missing_counts": dict(missing),
        "numeric": {
            "fps": fps_stats.summary(quantiles),
            "fps_from_frames": fps_from_frames_stats.summary(quantiles),
            "duration": duration_stats.summary(quantiles),
            "frames": frames_stats.summary(quantiles),
            "width": width_stats.summary(quantiles),
            "height": height_stats.summary(quantiles),
            "aspect_ratio": aspect_ratio_stats.summary(quantiles),
            "area": area_stats.summary(quantiles),
            "imaging_quality": imaging_quality_stats.summary(quantiles),
            "caption_len": caption_len_stats.summary(quantiles),
            "caption_words": caption_words_stats.summary(quantiles),
        },
        "topk": {
            "fps_rounded": _format_topk(fps_rounded, topk),
            "resolution": _format_topk(resolution, topk),
            "video_dir": _format_topk(video_dir, topk),
        },
        "consistency": {"fps_mismatch_count": fps_mismatch, "fps_mismatch_examples": fps_mismatch_examples},
    }


def print_text_summary(stats: Dict[str, Any], topk: int) -> None:
    print(f"Total records: {stats['total_records']}")
    if stats.get("missing_counts"):
        print("Missing counts (top 20):")
        for k, v in sorted(stats["missing_counts"].items(), key=lambda kv: kv[1], reverse=True)[:20]:
            print(f"  {k}: {v}")

    print("\nNumeric metrics:")
    for name, s in stats["numeric"].items():
        if s.get("count", 0) == 0:
            print(f"- {name}: count=0")
            continue
        q = s.get("quantiles", {})
        q_str = ""
        if q:
            keys = ["0.05", "0.5", "0.95"]
            picked = {k: q[k] for k in keys if k in q}
            if picked:
                q_str = f", q={picked}"
        print(
            f"- {name}: n={s['count']}, min={s['min']:.6g}, max={s['max']:.6g}, "
            f"mean={s['mean']:.6g}, std={s['std']:.6g}{q_str}"
        )

    print("\nTop-K distributions:")
    for k, items in stats["topk"].items():
        print(f"- {k} (top {topk}):")
        for row in items:
            print(f"  {row['value']}: {row['count']}")

    c = stats.get("consistency", {})
    if c:
        print(f"\nConsistency: fps_mismatch_count={c.get('fps_mismatch_count', 0)} (abs(fps - frames/duration) > 0.5)")
        for ex in c.get("fps_mismatch_examples", []):
            print(f"  example: {ex}")


def main():
    parser = argparse.ArgumentParser(description="统计 VidGen-1M metadata(json/jsonl) 的各类指标。")
    parser.add_argument(
        "--input",
        type=str,
        default="/mnt/vision-gen-ks3/Video_Generation/DataSets/vidgen-1M-sub/jsons/VidGen_1M_video3_recaption.jsonl",
        help="Path to metadata json/jsonl file.",
    )
    parser.add_argument("--format", type=str, default="text", choices=["text", "json"], help="Output format.")
    parser.add_argument("--round_digits", type=int, default=3, help="Round fps before counting frequencies.")
    parser.add_argument("--topk", type=int, default=10, help="How many most common buckets to show.")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100_000,
        help="Reservoir sample size used for quantiles (0 to disable quantiles).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reservoir sampling.")
    parser.add_argument(
        "--quantiles",
        type=str,
        default="0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99",
        help="Comma-separated quantiles to report (e.g. 0.05,0.5,0.95).",
    )
    args = parser.parse_args()

    quantiles = []
    for s in args.quantiles.split(","):
        s = s.strip()
        if not s:
            continue
        try:
            quantiles.append(float(s))
        except ValueError:
            continue

    records, errors = iter_records(args.input)
    stats = compute_stats(
        records=records,
        round_digits=args.round_digits,
        topk=args.topk,
        sample_size=args.sample_size,
        seed=args.seed,
        quantiles=quantiles,
    )
    stats["load_errors"] = errors

    if args.format == "json":
        print(json.dumps(stats, ensure_ascii=False))
    else:
        print_text_summary(stats, args.topk)
        if errors:
            print("\nLoad errors:", errors)


if __name__ == "__main__":
    main()
