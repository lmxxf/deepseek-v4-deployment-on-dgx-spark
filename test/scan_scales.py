"""Scan E8M0 scale distributions across all 43 layers (pure disk read)."""
import json
import struct
import sys
from collections import defaultdict

IDX = "/home/lmxxf/work/deepseek-v4-flash-deployment/deepseek-v4-flash/model.safetensors.index.json"
DIR = "/home/lmxxf/work/deepseek-v4-flash-deployment/deepseek-v4-flash"

with open(IDX) as f:
    wmap = json.load(f)["weight_map"]

# group scale tensors by layer
by_file = defaultdict(list)
for name, fn in wmap.items():
    if ".ffn.experts." in name and name.endswith(".scale"):
        by_file[fn].append(name)

headers = {}

def get_header(fn):
    if fn not in headers:
        with open(f"{DIR}/{fn}", "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            headers[fn] = (hl, json.loads(f.read(hl)))
    return headers[fn]

import re
stats = defaultdict(lambda: [256, 0])  # layer -> [min, max] scale byte
proj_stats = defaultdict(lambda: [256, 0])  # (layer, proj) -> [min,max]

for fn, names in sorted(by_file.items()):
    hl, meta = get_header(fn)
    with open(f"{DIR}/{fn}", "rb") as f:
        for name in names:
            item = meta[name]
            begin, end = item["data_offsets"]
            f.seek(8 + hl + begin)
            data = f.read(end - begin)
            mn, mx = min(data), max(data)
            m = re.match(r"layers\.(\d+)\.ffn\.experts\.\d+\.(w\d)\.scale", name)
            if m is None:
                continue
            layer, proj = int(m.group(1)), m.group(2)
            s = stats[layer]
            s[0] = min(s[0], mn); s[1] = max(s[1], mx)
            p = proj_stats[(layer, proj)]
            p[0] = min(p[0], mn); p[1] = max(p[1], mx)

print("layer  min..max scale byte (e8m0: value=2^(b-127))")
for layer in sorted(stats):
    mn, mx = stats[layer]
    per = " ".join(f"{p}:{proj_stats[(layer,p)][0]}..{proj_stats[(layer,p)][1]}"
                   for p in ("w1", "w2", "w3"))
    flag = " <<<" if mx >= 142 else ""  # 2^15: fp4 6*2^15 close to fp16 max
    print(f"L{layer:2d}  {mn}..{mx}   {per}{flag}")
