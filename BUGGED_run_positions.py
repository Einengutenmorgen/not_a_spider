"""Path 2: probe every token position t (absolute, counted from 0), a few positions at a time.

    PLAN_ONLY=1 ACTS=/media/data/hau/acts_local python3 run_positions.py   # plan.csv, then stop

    CUDA_VISIBLE_DEVICES=3 ACTS=/media/data/hau/acts_local HF_HOME=/media/data/hau/hf \
      nohup python3 -u run_positions.py > logs/positions_$(date +%m%d_%H%M).log 2>&1 &

Parallel: one process per GPU, SHARD=i of N_SHARDS=n; process i takes every n-th position
(t % n == i). Each needs its own ~115 GiB of disk and ~15 GB of host RAM for the parked model.

Per pass of POS_PER_PASS positions:
  1. extract: one forward pass over the teacher-forced input_ids from positions.jsonl, cut
     after the last position of the pass, writes $ACT_DIR/tNNN/hLL.npy (one row per case that
     reaches t) and rows.npy (their indices into data.analysis_rows). The states are kept
     bit-exact in the model's native bfloat16, as raw bits in int16 (numpy has no bfloat16).
  2. probe: every feature (h00..h31, d01..d31) in every condition of data.conditions
     (ident, wording, frame, slot/0-2, strict/0-2), restricted to the cases at t;
     standardise on train, no PCA, logistic regression C=1, L-BFGS as in gpu_probe.py,
     train size matched across conditions within the position
     -> $OUT/results/tNNN.csv and $OUT/preds/tNNN.npz (test predictions, every feature)
  3. delete $ACT_DIR/tNNN; $OUT is copied to RESULTS_DEST with rclone (RESULTS_DEST="" skips)

A case enters position t only from its start_of_ident on: system turn, user header and the
question text before the identifier are not tested; positions no case reaches are marked
not tested in plan.csv. Comparison by function (newline, answer, ...) comes later by joining
preds (uids.npy[rows], t) with tokens.parquet. Activations can be recreated for single
positions with extract_positions.py.

Resumable: rerun the same command. Probed positions are skipped, extracted ones are not
extracted again.
"""
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

import data

MODEL = os.environ.get("MODEL", "allenai/Olmo-3-7B-Instruct")
ACT_DIR = Path(os.environ.get("ACT_DIR", "/media/data/hau/acts_t"))
OUT = Path(os.environ.get("OUT", "results_positions"))
RESULTS_DEST = os.environ.get("RESULTS_DEST", "NotaSpider:not_a_spider/positions/results")
DEVICE = os.environ.get("DEVICE", "cuda")
POS_PER_PASS = int(os.environ.get("POS_PER_PASS", 2))
FEAT_CHUNK = int(os.environ.get("FEAT_CHUNK", 2))    # features fitted together; GPU ~16 GB at 2
SHARD, N_SHARDS = int(os.environ.get("SHARD", 0)), int(os.environ.get("N_SHARDS", 1))
BATCH, MIN_TRAIN, SEED, C = 64, 200, 0, 1.0

# ---------- cases, splits, positions ----------
_, tab = data.analysis_rows(data.build_table())
uids = np.array([r["uid"] for r in tab])
classes, y = np.unique([r["animal"] for r in tab], return_inverse=True)
conds = data.conditions(tab, seed=SEED)
assert all(len(idx["dev"]) == 0 for _, _, idx in conds), \
    "data.py still holds out dev identifiers: holdout_identifiers(tab, n_test=4, n_dev=0, ...)"

pos = {}
for line in open("positions.jsonl"):
    r = json.loads(line)
    pos[r["uid"]] = (r["input_ids"], r["start_of_ident"])
ids = [pos[u][0] for u in uids]
start = np.array([pos[u][1] for u in uids])
lens = np.array([len(x) for x in ids])
del pos

cfg = AutoConfig.from_pretrained(MODEL)
L, D = cfg.num_hidden_layers, cfg.hidden_size                       # h00..h31 as in Path 1
reach = [np.nonzero((start <= t) & (lens > t))[0] for t in range(lens.max())]
(OUT / "results").mkdir(parents=True, exist_ok=True)
(OUT / "preds").mkdir(exist_ok=True)
ACT_DIR.mkdir(parents=True, exist_ok=True)
if SHARD == 0:                                                      # shared files: one writer
    np.save(OUT / "uids.npy", uids)                                 # preds index into this
    with open(OUT / "plan.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "n_rows", "gib", "tested"])
        w.writerows([t, len(r), f"{len(r) * L * D * 2 / 2**30:.1f}", len(r) > 0] for t, r in enumerate(reach))
tested = [t for t, r in enumerate(reach) if len(r)]
mine = [t for t in tested if t % N_SHARDS == SHARD]
print(f"{len(uids)} cases, conditions {[c for c, _, _ in conds]} | positions t{tested[0]}..t{tested[-1]} "
      f"tested, t0..t{tested[0] - 1} not | largest position "
      f"{max(len(r) for r in reach) * L * D * 2 / 2**30:.0f} GiB", flush=True)
if os.environ.get("PLAN_ONLY"):
    raise SystemExit


def act_dir(t):
    return ACT_DIR / f"t{t:03d}"


def probed(t):
    return (OUT / "results" / f"t{t:03d}.csv").exists()


def extracted(t):
    return (act_dir(t) / "complete").exists()


# ---------- extract ----------
def extract(model, ts):
    gib = sum(len(reach[t]) for t in ts) * L * D * 2 / 2**30
    free = shutil.disk_usage(ACT_DIR).free / 2**30
    assert gib * N_SHARDS + 2 < free, \
        f"t{ts}: {N_SHARDS} process(es) need {gib * N_SHARDS:.0f} GiB, {free:.0f} GiB free on {ACT_DIR}"
    out, slot = {}, {}
    for t in ts:
        act_dir(t).mkdir(exist_ok=True)
        np.save(act_dir(t) / "rows.npy", reach[t])
        slot[t] = np.full(len(uids), -1)
        slot[t][reach[t]] = np.arange(len(reach[t]))
        out[t] = [np.lib.format.open_memmap(act_dir(t) / f"h{k:02d}.npy", mode="w+", dtype=np.int16,
                                            shape=(len(reach[t]), D)) for k in range(L)]
    cut = max(ts) + 1                                                # nothing after t is needed
    rows = np.unique(np.concatenate([reach[t] for t in ts]))
    rows = rows[np.argsort(np.minimum(lens[rows], cut), kind="stable")]
    pad = model.config.pad_token_id or 0
    for s in range(0, len(rows), BATCH):
        b = rows[s:s + BATCH]
        seqs = [ids[i][:cut] for i in b]
        x = torch.full((len(b), max(map(len, seqs))), pad)
        mask = torch.zeros_like(x)
        for j, q in enumerate(seqs):                                 # right padding
            x[j, :len(q)], mask[j, :len(q)] = torch.tensor(q), 1
        with torch.inference_mode():
            hs = model(input_ids=x.to(DEVICE), attention_mask=mask.to(DEVICE),
                       output_hidden_states=True).hidden_states[:L]
        for t in ts:
            ok = np.nonzero(slot[t][b] >= 0)[0]
            if len(ok):
                st = torch.stack([h[torch.as_tensor(ok, device=DEVICE), t] for h in hs])
                assert st.dtype == torch.bfloat16 and torch.isfinite(st).all()
                st = st.view(torch.int16).cpu().numpy()                     # raw bfloat16 bits
                for k in range(L):
                    out[t][k][slot[t][b[ok]]] = st[k]
    for t in ts:
        for m in out[t]:
            m.flush()
        (act_dir(t) / "complete").touch()


# ---------- probe ----------
def fit(Z, yy, n_classes):
    """Independent logistic regressions, one per Z[f], summed losses in one L-BFGS run.
    Same objective per problem as gpu_probe.py: mean CE + ||W||^2 / (2 C n)."""
    nf, m, d = Z.shape
    W = torch.zeros(nf, d, n_classes, device=DEVICE, requires_grad=True)
    b = torch.zeros(nf, 1, n_classes, device=DEVICE, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=2000, history_size=20, tolerance_grad=1e-5,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")
    lam, yy = 1 / (2 * C * m), yy.repeat(nf)

    def closure():
        opt.zero_grad()
        loss = nf * F.cross_entropy(torch.baddbmm(b, Z, W).reshape(-1, n_classes), yy) + lam * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach(), opt.state[W]["n_iter"]


def probe(t):
    rows = np.load(act_dir(t) / "rows.npy")
    present = np.zeros(len(uids), bool)
    present[rows] = True
    slot = np.full(len(uids), -1)
    slot[rows] = np.arange(len(rows))
    parts = [(name, idx["train"][present[idx["train"]]], idx["test"][present[idx["test"]]])
             for name, _, idx in conds]
    n_match = min(len(tr) for _, tr, _ in parts)                    # same train size in every condition
    feats = [(f"h{k:02d}", k, None) for k in range(L)] + [(f"d{k:02d}", k, k - 1) for k in range(1, L)]
    res, preds = [], {}
    for s in range(0, len(feats), FEAT_CHUNK):
        chunk = feats[s:s + FEAT_CHUNK]
        H = {k: torch.from_numpy(np.load(act_dir(t) / f"h{k:02d}.npy")).to(DEVICE).view(torch.bfloat16)
             for k in {k for _, k, _ in chunk} | {j for _, _, j in chunk if j is not None}}
        X = torch.stack([H[k].float() - (H[j].float() if j is not None else 0) for _, k, j in chunk])
        del H
        for name, tr, te in parts:
            if n_match < MIN_TRAIN or len(np.unique(y[te])) < 2:
                res += [[t, f, name, n_match, len(te), "", "skipped"] for f, _, _ in chunk]
                continue
            tr = np.asarray(data.match_size(tr, y, n_match, seed=SEED))
            Z = X[:, torch.as_tensor(slot[np.concatenate([tr, te])], device=DEVICE)]
            mu = Z[:, :len(tr)].mean(1, keepdim=True)
            sd = Z[:, :len(tr)].std(1, unbiased=False, keepdim=True)
            sd[sd == 0] = 1
            Z.sub_(mu).div_(sd)
            W, b, it = fit(Z[:, :len(tr)], torch.as_tensor(y[tr], device=DEVICE), len(classes))
            pred = torch.baddbmm(b, Z[:, len(tr):], W).argmax(-1)                     # [features, test]
            acc = (pred == torch.as_tensor(y[te], device=DEVICE)).float().mean(1)
            res += [[t, f, name, len(tr), len(te), f"{a:.4f}", it] for (f, _, _), a in zip(chunk, acc.tolist())]
            preds.setdefault(name, [te]).append(pred.to(torch.int8).cpu().numpy())
            del Z
        del X
    torch.cuda.empty_cache()
    np.savez_compressed(OUT / "preds" / f"t{t:03d}.npz", classes=classes, features=[f for f, _, _ in feats],
                        **{f"{n.replace('/', '_')}_rows": p[0] for n, p in preds.items()},
                        **{f"{n.replace('/', '_')}_pred": np.concatenate(p[1:]) for n, p in preds.items()})
    tmp = OUT / "results" / f"t{t:03d}.tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "feature", "condition", "n_train", "n_test", "acc", "n_iter"])
        w.writerows(res)
    tmp.replace(OUT / "results" / f"t{t:03d}.csv")                  # written last: marks t as probed


def backup():
    if RESULTS_DEST:
        r = subprocess.run(["rclone", "copy", str(OUT), RESULTS_DEST], capture_output=True, text=True)
        if r.returncode:
            print(f"results backup failed (kept locally): {r.stderr.strip()[-300:]}", flush=True)


# ---------- run ----------
for t in mine:                                      # left over from an interrupted run
    if probed(t) and act_dir(t).exists():
        shutil.rmtree(act_dir(t))
model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
todo = [t for t in mine if not probed(t)]
print(f"shard {SHARD}/{N_SHARDS}: {len(todo)} positions to do", flush=True)
for p in range(0, len(todo), POS_PER_PASS):
    ts = todo[p:p + POS_PER_PASS]
    t0 = time.time()
    if missing := [t for t in ts if not extracted(t)]:
        model.to(DEVICE)
        extract(model, missing)
        model.to("cpu")                                              # free the GPU for probing
    t1 = time.time()
    for t in ts:
        probe(t)
        shutil.rmtree(act_dir(t))
    backup()
    print(f"t{ts[0]:03d}-t{ts[-1]:03d}  extract {t1 - t0:5.0f}s  probe {time.time() - t1:5.0f}s  "
          f"[{p + len(ts)}/{len(todo)}]", flush=True)