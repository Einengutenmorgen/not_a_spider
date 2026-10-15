"""Hidden states at named prompt positions for every analysis row, for gpu_probe.py --coords.

    CUDA_VISIBLE_DEVICES=3 ACTS=/media/data/hau/acts_local HF_HOME=/media/data/hau/hf \
      nohup python3 -u extract_positions.py end_of_ident end_of_ident+1 end_user_formatting-1 \
        newline first_assistant_temp > logs/extract_$(date +%m%d_%H%M).log 2>&1 &

A position is an anchor from positions.jsonl, optionally shifted by a token count:
end_of_ident+1 is the token after the identifier (post+0 in the pre-test) and
end_user_formatting-1 the last question token (post-1). Only prompt positions are allowed:
the forward pass runs on the prompt alone, which by causality gives the same states.

Writes $POS_DIR/<position>/hNN.npy, float16, one row per data.analysis_rows row in the
same order. Disk: about 61 GB per position (32 layers x 233,900 rows x 4096).

Resumable: every CKPT batches the files are flushed and $POS_DIR/<position>/done.txt records
how many rows (in length order) are complete. Rerun the same command after a crash; it
checks free disk space against what is still unwritten before starting.
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

import data

MODEL = os.environ.get("MODEL", "allenai/Olmo-3-7B-Instruct")
POS_DIR = Path(os.environ.get("POS_DIR", "/media/data/hau/acts_pos"))
DEVICE = os.environ.get("DEVICE", "cuda")
BATCH = 64
CKPT = 200              # batches between checkpoints
NAMES = sys.argv[1:]


def resolve(name, r):
    anchor, shift = re.fullmatch(r"([a-z_]+)([+-]\d+)?", name).groups()
    t = r[anchor] + int(shift or 0)
    assert 0 <= t < r["n_prompt"], f"{name} lies outside the prompt in {r['uid']}"
    return t


_, tab = data.analysis_rows(data.build_table())
pos = {r["uid"]: r for r in map(json.loads, open("positions.jsonl"))}
rows = [pos[r["uid"]] for r in tab]
n = len(rows)
T = {name: np.array([resolve(name, r) for r in rows]) for name in NAMES}

model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEVICE).eval()
L, D = model.config.num_hidden_layers, model.config.hidden_size         # h00..h31 as in Path 1
out, done = {}, {}
for name in NAMES:
    (POS_DIR / name).mkdir(parents=True, exist_ok=True)
    marker = POS_DIR / name / "done.txt"
    done[name] = int(marker.read_text()) if marker.exists() else 0
    for k in range(L):
        path = POS_DIR / name / f"h{k:02d}.npy"
        out[name, k] = (np.lib.format.open_memmap(path, mode="r+") if path.exists() else
                        np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, D)))
        assert out[name, k].shape == (n, D), f"{path} has shape {out[name, k].shape}"

# files are sparse: blocks are allocated as rows get written, so check what is still missing
need = sum(max(0, m.nbytes - Path(m.filename).stat().st_blocks * 512) for m in out.values())
free = shutil.disk_usage(POS_DIR).free
print(f"resume from {done} rows | still to allocate {need / 1e9:.0f} GB, free {free / 1e9:.0f} GB")
assert need < free, "not enough disk space"


def checkpoint(rows_done):
    for name in NAMES:
        if rows_done > done[name]:
            for k in range(L):
                out[name, k].flush()
            tmp = POS_DIR / name / "done.tmp"
            tmp.write_text(str(rows_done))
            tmp.replace(POS_DIR / name / "done.txt")           # atomic: never a half-written marker
            done[name] = rows_done
    print(f"{rows_done}/{n}", flush=True)


pad = model.config.pad_token_id or 0
by_len = np.argsort([r["n_prompt"] for r in rows], kind="stable")     # same order on every run
for s in range(min(done.values()) // BATCH * BATCH, n, BATCH):
    b = by_len[s:s + BATCH]
    todo = [name for name in NAMES if s >= done[name]]
    seqs = [rows[i]["input_ids"][:rows[i]["n_prompt"]] for i in b]
    ids = torch.full((len(b), max(map(len, seqs))), pad)
    mask = torch.zeros_like(ids)
    for j, q in enumerate(seqs):                                        # right padding
        ids[j, :len(q)], mask[j, :len(q)] = torch.tensor(q), 1
    with torch.inference_mode():
        hs = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE),
                   output_hidden_states=True).hidden_states[:L]
    rng = torch.arange(len(b), device=DEVICE)
    for name in todo:
        t = torch.as_tensor(T[name][b], device=DEVICE)
        st = torch.stack([h[rng, t] for h in hs]).to(torch.float16)
        assert torch.isfinite(st).all(), "float16 overflow"
        st = st.cpu().numpy()
        for k in range(L):
            out[name, k][b] = st[k]
    if (s // BATCH + 1) % CKPT == 0 or s + len(b) == n:
        checkpoint(s + len(b))
print(f"-> {POS_DIR}: {', '.join(NAMES)}")