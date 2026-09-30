"""Which token sits at the readout position? h00 is the input embedding, so every
distinct h00 vector is one distinct token. Expected: exactly one (the '\\n' after
'<|im_start|>assistant'). More than one means some rows were read elsewhere.

    ACTS=/media/data/hau/acts_local HF_HOME=/media/data/hau/hf python3 position_check.py
"""
import collections
import json
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

import data

MODEL = os.environ.get("MODEL", "allenai/Olmo-3-7B-Instruct")

tab = data.build_table()                        # all rows, before any drop list
rows = np.arange(len(tab))
H0 = data.feature("h00", rows)
print("row fields:", sorted(tab[0]))

# group rows by their exact h00 vector
groups = collections.defaultdict(list)
for i, v in enumerate(H0):
    groups[v.tobytes()].append(i)
groups = sorted(groups.values(), key=len, reverse=True)
print(f"\n{len(groups)} distinct h00 vectors over {len(tab)} rows\n")

# map each vector to its nearest row of the embedding matrix
tok = AutoTokenizer.from_pretrained(MODEL)
index = json.load(open(hf_hub_download(MODEL, "model.safetensors.index.json")))
key = next(k for k in index["weight_map"] if k.endswith("embed_tokens.weight"))
with safe_open(hf_hub_download(MODEL, index["weight_map"][key]), framework="pt") as fh:
    E = fh.get_tensor(key).float()

for g in groups[:20]:
    v = torch.as_tensor(H0[g[0]])
    dist = (E - v).norm(dim=1)
    tid = int(dist.argmin())
    rel = dist[tid].item() / v.norm().item()
    sub = [tab[i] for i in g]
    top = lambda f: collections.Counter(r.get(f) for r in sub).most_common(3)
    print(f"n={len(g):6d}  token {tid} {tok.decode([tid])!r}  rel.dist {rel:.1e}  "
          f"animals {len({r['animal'] for r in sub})}")
    print(f"          frame {top('frame')}  slot {top('slot')}  n_tokens {min(r['n_tokens'] for r in sub)}"
          f"-{max(r['n_tokens'] for r in sub)}")
    if "pos_tokens" in sub[0]:        # what the extraction itself recorded for this position
        pt = collections.Counter(r["pos_tokens"].get("prompt_last") for r in sub).most_common(3)
        print(f"          recorded pos_tokens[prompt_last] {pt}")
    for r in sub[:2]:
        print(f"          e.g. ...{r.get('question', r.get('user', ''))[-70:]!r}")
if len(groups) > 20:
    print(f"... {len(groups) - 20} more groups, {sum(map(len, groups[20:]))} rows")
