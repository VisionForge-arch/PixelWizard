import os
import re
import difflib
import json

# === 配置路径 ===
prompt_txt = "/root/ultrawan/Wan2.2/prompt.txt"                   # 每行一个 prompt
video_dir  = "/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/480p_5s/decode_video"        # 存放 .mp4 或 .pt 文件的目录


# ===== 命名规则 =====
prefix = "ti2v-5B_832*480_8_+"
pattern = re.compile(r"^" + re.escape(prefix) + r"(.+?)_\d{8}_\d{6}\.pt$")

# ===== 读取 prompts =====
with open(prompt_txt, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# ===== 解析文件名中的 prompt 片段 =====
files = [f for f in os.listdir(video_dir) if f.endswith(".pt") or f.endswith(".mp4")]
file_prompts = {}
for f in files:
    m = pattern.match(f)
    if m:
        frag = m.group(1)
        file_prompts[f] = frag

# ===== 规范化函数（与保存时完全一致） =====
def normalize_prompt(p):
    return p.replace(" ", "_").replace("/", "_")[:50]

# ===== 建立映射 =====
mapping = []
for p in prompts:
    norm_p = normalize_prompt(p)
    # 精确匹配
    matched = next((fn for fn, frag in file_prompts.items() if frag == norm_p), None)
    # 如果找不到，就模糊匹配一次
    if not matched:
        best = difflib.get_close_matches(norm_p, file_prompts.values(), n=1, cutoff=0.7)
        if best:
            matched = next(fn for fn, frag in file_prompts.items() if frag == best[0])
    mapping.append({
        "prompt": p,
        "file": os.path.join(video_dir, matched) if matched else None
    })

# ===== 保存结果 =====
out_json = "prompt_to_file.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)

print(f"✅ 共匹配 {len(mapping)} 条，结果已保存到 {out_json}")