import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict


def load_prompts(path: Path) -> List[str]:
    """读取文本文件并返回去除空行的条目列表。"""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def build_records(prompts: List[str], start_id: int = 1) -> List[Dict]:
    """为每条 prompt 编号。"""
    records = []
    for idx, text in enumerate(prompts, start=start_id):
        records.append(
            {
                "id": idx,
                "text": text,
            }
        )
    return records


def save_jsonl(records: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_csv(records: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(
        description="将 extracted_texts.txt 整理成带编号的 jsonl/csv，方便评测与调用。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../extracted_texts.txt"),
        help="输入文本文件路径，每行一条。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/prompts.jsonl"),
        help="输出路径，后缀不限制（由 --format 决定内容）。",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="输出文件格式。",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="编号起始值，便于与其他数据集衔接。",
    )

    args = parser.parse_args()

    prompts = load_prompts(args.input)
    if not prompts:
        raise SystemExit(f"未从 {args.input} 读取到有效内容")

    records = build_records(prompts, start_id=args.start_id)

    if args.format == "jsonl":
        save_jsonl(records, args.output)
    else:
        save_csv(records, args.output)

    print(f"已处理 {len(records)} 条，保存到 {args.output}")


if __name__ == "__main__":
    main()
