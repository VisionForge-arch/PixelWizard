import os
import re
import difflib
import json

# === 配置路径 ===
i2v_json  = "/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/I2V/i2v_vbench.json"  # JSON 中包含 caption/frame
video_dir = "/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/i2v/480p/decode_video"        # 存放 .mp4 或 .pt 文件的目录


pat = re.compile(r"^ti2v-5B_832\*480_8_(?:\+?prompt_)?(?P<frag>.+?)_\d{8}_\d{6}\.(?:pt|mp4)$")

def normalize(p: str) -> str:
    # 和你保存时完全一致：只做空格与斜杠替换，然后截断 50
    return p.replace(" ", "_").replace("/", "_")[:50]

# 读取 JSON 数据
with open(i2v_json, "r", encoding="utf-8") as f:
    raw_entries = json.load(f)

prompts = [
    {
        "caption": (entry.get("caption") or "").strip(),
        "frame_path": entry.get("frame_path"),
    }
    for entry in raw_entries
    if entry.get("caption")
]

# 收集目录中的文件片段
files = [f for f in os.listdir(video_dir) if f.endswith((".pt", ".mp4"))]
file_frag_by_name = {}
frags = []
for f in files:
    m = pat.match(f)
    if m:
        frag = m.group("frag")
        file_frag_by_name[f] = frag
        frags.append(frag)

mapping = []
unmatched = 0
for entry in prompts:
    p = entry["caption"]
    np = normalize(p)
    # 先精确匹配
    exact = next((fn for fn, frag in file_frag_by_name.items() if frag == np), None)
    if exact is None:
        # 退化为相似度匹配（避免标点等轻微差异）
        cand = difflib.get_close_matches(np, frags, n=1, cutoff=0.75)
        if cand:
            exact = next(fn for fn, frag in file_frag_by_name.items() if frag == cand[0])
    if exact is None:
        unmatched += 1
        continue

    mapping.append({
        "prompt": p,
        "normalized": np,
        "frame_path": entry["frame_path"],
        "file": os.path.join(video_dir, exact)
    })

out_json = "/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/480p_i2v_matched.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)

print(
    f"✅ 完成 {len(mapping)} 条匹配，跳过 {unmatched} 条未匹配，结果已写入 {out_json}"
)