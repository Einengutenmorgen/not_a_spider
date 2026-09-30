"""Download an extraction run from Drive and re-lay it out layer-major.

    python fetch_layers.py NotaSpider:not_a_spider/full_run acts_local
    python fetch_layers.py NotaSpider:not_a_spider/full_run acts_local --layers 16 20 21 22 24

Shards hold [rows, 33 layers, 4096], but a probe reads one layer across all rows. Each
shard is downloaded, written into h{kk}.npy (one float16 array per layer, row i = line i
of rows.jsonl), then deleted. Peak extra disk is one shard (~0.5 GiB); each layer file is
~1.8 GiB, all 33 ~59 GiB. Rerunning resumes; keep --layers the same across reruns.
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

RCLONE = str(Path.home() / "bin" / "rclone")

ap = argparse.ArgumentParser()
ap.add_argument("remote")
ap.add_argument("out", type=Path)
ap.add_argument("--layers", type=int, nargs="*", help="hidden-state indices; default all")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

subprocess.run([RCLONE, "copy", a.remote, str(a.out),
                "--include", "index_*.jsonl", "--include", "meta.json"], check=True)
rows = [json.loads(l) for p in sorted(a.out.glob("index_*.jsonl"))
        for l in p.read_text().splitlines()]
assert len({r["uid"] for r in rows}) == len(rows), "duplicate uids across index files"
with (a.out / "rows.jsonl").open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
N = len(rows)

where = {}   # shard -> (row in shard, row in layer file)
for pos, r in enumerate(rows):
    where.setdefault(r["shard"], []).append((r["row"], pos))

done_f = a.out / "shards_done.txt"
done = set(done_f.read_text().split()) if done_f.exists() else set()
mm = {}
for n, (shard, pairs) in enumerate(where.items(), 1):
    if shard in done:
        continue
    subprocess.run([RCLONE, "copy", f"{a.remote}/{shard}", str(a.out)])
    path = a.out / shard
    if not path.exists():
        raise SystemExit(f"{shard} is in the index but not on {a.remote} -- was it left "
                         f"local on the extraction machine?")
    t = load_file(str(path))["prompt_last"].float()
    assert t.abs().max() < 6e4, f"{shard}: values outside float16 range"
    src, dst = map(np.array, zip(*pairs))
    x = t[torch.as_tensor(src)].half().numpy()
    for k in a.layers if a.layers is not None else range(t.shape[1]):
        if k not in mm:
            f = a.out / f"h{k:02d}.npy"
            mm[k] = np.lib.format.open_memmap(f, mode="r+" if f.exists() else "w+",
                                              dtype=np.float16, shape=(N, t.shape[2]))
        mm[k][dst] = x[:, k]
    for m in mm.values():
        m.flush()
    path.unlink()
    with done_f.open("a") as f:
        f.write(shard + "\n")
    print(f"{n}/{len(where)} {shard}", flush=True)

print(f"done: {N} rows, layers {sorted(mm) or 'already complete'} -> {a.out}")
