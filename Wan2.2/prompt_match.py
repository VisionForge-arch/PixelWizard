import os
import re
import difflib
import json

# === 配置路径 ===
prompt_txt = "/root/ultrawan/Wan2.2/prompt.txt"                   # 每行一个 prompt
video_dir  = "/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/Output/outputs_ultra/480p_base/360p_5s/decoded_video_3900"        # 存放 .mp4 或 .pt 文件的目录


pat = re.compile(r"^ti2v-5B_640\*352_1_(?:\+?prompt_)?(?P<frag>.+?)_\d{8}_\d{6}\.(?:pt|mp4)$")

def normalize(p: str) -> str:
    # 和你保存时完全一致：只做空格与斜杠替换，然后截断 50
    return p.replace(" ", "_").replace("/", "_")[:50]

# 读取 prompts
with open(prompt_txt, "r", encoding="utf-8") as f:
    prompts = [ln.strip() for ln in f if ln.strip()]

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
for p in prompts:
    np = normalize(p)
    # 先精确匹配
    exact = next((fn for fn, frag in file_frag_by_name.items() if frag == np), None)
    if exact is None:
        # 退化为相似度匹配（避免标点等轻微差异）
        cand = difflib.get_close_matches(np, frags, n=1, cutoff=0.75)
        if cand:
            exact = next(fn for fn, frag in file_frag_by_name.items() if frag == cand[0])
    # 只有匹配成功时才添加到 mapping
    if exact is not None:
        mapping.append({
            "prompt": p,
            "normalized": np,
            "file": os.path.join(video_dir, exact)
        })

out_json = "/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/prompt_to_file_360p.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)

print(f"✅ 完成 {len(mapping)} 条匹配，结果已写入 {out_json}")