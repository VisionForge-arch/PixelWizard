import json, random, os

src = "/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/UltraVideo/matched_short.json"
dst = "/mnt/vision-gen-ks3/IndividualDirs/zp/wenxueli/UltraVideo/matched_short_16.json"

with open(src, "r") as f:
    data = json.load(f)

# 选择其中一种：前16 or 随机16
subset = data[:16]
# subset = random.sample(data, k=min(16, len(data)))

with open(dst, "w") as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)

print(f"Saved {len(subset)} items to {dst}")