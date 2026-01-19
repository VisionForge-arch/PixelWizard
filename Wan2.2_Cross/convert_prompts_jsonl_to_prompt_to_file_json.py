#!/usr/bin/env python3
import argparse
import json
import os
import sys


PROMPT_FALLBACK_KEYS = ("prompt", "text", "caption", "detailed_description", "description")
FILE_FALLBACK_KEYS = ("file", "video_path", "path", "file_path", "filepath")
ID_FALLBACK_KEYS = ("id", "clip_id")


def normalize_prompt(prompt: str) -> str:
    return prompt.replace(" ", "_").replace("/", "_")[:50]


def _read_jsonl(path: str, encoding: str):
    with open(path, "r", encoding=encoding) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}, got {type(obj).__name__}")
            yield line_no, obj


def _pick_first_str(d: dict, keys: tuple[str, ...]):
    for k in keys:
        v = d.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip(), k
    return None, None


def _pick_first_non_empty(d: dict, keys: tuple[str, ...]):
    for k in keys:
        v = d.get(k, None)
        if v is None:
            continue
        if isinstance(v, str):
            v = v.strip()
            if not v:
                continue
        return v, k
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Convert prompts JSONL (e.g. {id,text}) to Wan-compatible JSON list (e.g. {prompt,file})."
    )
    parser.add_argument(
        "-i",
        "--input",
        #default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/prompts.jsonl",
        default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/prompt_vbench.jsonl",
        help="Input JSONL path, one JSON object per line.",
    )
    parser.add_argument(
        "-o",
        "--output",
        #default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/eval_100_match_240_new_8000.json",
        default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompts/eval_vbench_match_240_1.json",
        help="Output JSON path (list of objects).",
    )
    parser.add_argument(
        "--pt-dir",
        #default="/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/eval_100/240p_5s/pt_new_5000",
        default="/mnt/nas01-ak/IndividualDirs/wenxueli/eval_vbench/240p_5s_2/seed_1/pt",
        help="Directory to build file path from id/clip_id when 'file' is missing.",
    )
    parser.add_argument(
        "--ext",
        default=".pt",
        help="Extension appended when building file path from id (default: .pt).",
    )
    parser.add_argument(
        "--add-normalized",
        default=False, 
        help="Also add 'normalized' field like prompt_match.py does.",
    )
    parser.add_argument(
        "--keep-id",
        action="store_true",
        help="Keep original id/clip_id field in output objects.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any bad/missing record instead of skipping.",
    )
    args = parser.parse_args()

    ext = args.ext if args.ext.startswith(".") else f".{args.ext}"
    out = []
    skipped = 0

    for line_no, item in _read_jsonl(args.input, args.encoding):
        prompt, _ = _pick_first_str(item, PROMPT_FALLBACK_KEYS)
        file_path, _ = _pick_first_non_empty(item, FILE_FALLBACK_KEYS)

        id_value, id_key = _pick_first_non_empty(item, ID_FALLBACK_KEYS)

        if file_path is None and id_value is not None:
            file_id = str(id_value).strip()
            base = os.path.basename(file_id)
            if "." not in base:
                file_id = f"{file_id}{ext}"
            file_path = os.path.join(args.pt_dir, file_id)

        if not prompt or not file_path:
            skipped += 1
            msg = f"skip {args.input}:{line_no} missing prompt/file (got prompt={bool(prompt)} file={bool(file_path)})"
            if args.strict:
                raise ValueError(msg)
            print(msg, file=sys.stderr)
            continue

        rec = {"prompt": prompt, "file": str(file_path)}
        if args.add_normalized:
            rec["normalized"] = normalize_prompt(prompt)
        if args.keep_id and id_key is not None:
            rec[id_key] = id_value
        out.append(rec)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding=args.encoding) as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"ok: wrote {len(out)} records to {args.output} (skipped {skipped})")


if __name__ == "__main__":
    main()

