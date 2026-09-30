"""Path 2 pre-test: probe every token position on 5 animals.

    CUDA_VISIBLE_DEVICES=3 ACTS=/media/data/hau/acts_local HF_HOME=/media/data/hau/hf \
      nohup python3 -u pretest_positions.py > logs/pretest_$(date +%m%d_%H%M).log 2>&1 &

Reads positions.jsonl (index_positions.py) and runs one teacher-forced forward pass per row,
no generation. Tokens are aligned across rows by segment coordinates:

    sys-1          last template token before the question (identical in every row: chance)
    pre, id, post  question text before / inside / after the identifier wording
    tmpl+0..+4     <|im_end|> \\n <|im_start|> assistant \\n   (tmpl+4 = Path 1's position)
    ans            answer tokens    only rows whose answer is complete and names no animal
    close          <|im_end|> after the answer, same rows

Variable segments are indexed K tokens from each end: id+0 is start_of_ident, id-1 is
end_of_ident. coords.csv maps the anchors onto coordinates and lists the most common token.

Every coordinate is probed on h00..h31 and d01..d31 (d_k = h_k - h_(k-1)) in the first fold
of each condition; splits come from data.py, restricted to the 5 animals. The probe is the
one from gpu_probe.py: standardise on train, no PCA, logistic regression C=1, L-BFGS.
"""
import csv
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import data

MODEL = os.environ.get("MODEL", "allenai/Olmo-3-7B-Instruct")
ANIMALS = os.environ.get("ANIMALS", "Elephant,Giraffe,Penguin,Octopus,Platypus").split(",")
CONDITIONS = ("strict", "ident", "wording")
ALL_FOLDS = False       # True: every fold of each condition (about 3x the fitting time)
K = 6                   # tokens per variable segment, counted from each end
POS_PER_PASS = int(os.environ.get("POS_PER_PASS", 3))   # host RAM: ~13 GB per coordinate
FEAT_CHUNK = int(os.environ.get("FEAT_CHUNK", 8))       # probes fitted together on the GPU
BATCH = 64
MIN_TRAIN = 200
SEED, C = 0, 1.0
DEVICE = os.environ.get("DEVICE", "cuda")
SEG_ORDER = ["tmpl", "sys", "pre", "id", "post", "ans", "close"]    # tmpl first: repro check
ANCHORS = {"start_of_ident": "id+0", "end_of_ident": "id-1", "end_user_formatting": "tmpl+0",
           "newline": "tmpl+1", "first_assistant_temp": "tmpl+2",
           "first_assistant_token": "ans+0", "last_assistant_token": "ans-1",
           "last_assistant_temp": "close"}

# ---------- rows, splits ----------
tab_rows, tab = data.analysis_rows(data.build_table())
tab_rows = np.asarray(tab_rows)
pos = {r["uid"]: r for r in map(json.loads, open("positions.jsonl")) if r["animal"] in ANIMALS}
sel = np.array([i for i, r in enumerate(tab) if r["animal"] in ANIMALS])
rows = [pos[tab[i]["uid"]] for i in sel]
n = len(rows)
classes, y_np = np.unique([r["animal"] for r in rows], return_inverse=True)
assert len(classes) == len(ANIMALS), f"animals not found: {set(ANIMALS) - set(classes)}"

where = {int(t): k for k, t in enumerate(sel)}          # full-table index -> subset index
conds, seen = [], set()
for name, _, idx in data.conditions(tab, seed=SEED):
    cond = name.split("/")[0]
    if cond in CONDITIONS and (ALL_FOLDS or cond not in seen):
        seen.add(cond)
        conds.append((name, {p: np.array([where[int(i)] for i in idx[p] if int(i) in where])
                             for p in ("train", "test")}))
print(f"{n} rows, classes {list(classes)}, conditions {[c for c, _ in conds]}")

# ---------- coordinates ----------
animals = json.loads(Path("animals.json").read_text())
names = {x for a, v in animals.items() for x in [a, *v.get("alternative_naming", [])]}
names_animal = re.compile("|".join(r"\b" + re.escape(x) for x in sorted(names, key=len, reverse=True)), re.I)

TMPL = {r["prompt_last"] - r["end_user_formatting"] for r in rows}
assert len(TMPL) == 1, f"assistant header length varies: {TMPL}"
PROMPT_LAST = ANCHORS["prompt_last"] = f"tmpl+{TMPL.pop()}"     # tmpl+4 for Olmo 3

ids0 = rows[0]["input_ids"]
q0 = min(next((i for i, (a, b) in enumerate(zip(ids0, r["input_ids"])) if a != b), len(ids0))
         for r in rows)                                 # shared template prefix = sys segment


def coordinates(r):
    c = {"sys-1": q0 - 1}

    def seg(name, a, b):
        for k in range(min(K, b - a)):
            c[f"{name}+{k}"], c[f"{name}-{k + 1}"] = a + k, b - 1 - k

    assert r["start_of_ident"] >= q0
    seg("pre", q0, r["start_of_ident"])
    seg("id", r["start_of_ident"], r["end_of_ident"] + 1)
    seg("post", r["end_of_ident"] + 1, r["end_user_formatting"])
    for k in range(r["prompt_last"] - r["end_user_formatting"] + 1):
        c[f"tmpl+{k}"] = r["end_user_formatting"] + k
    if r["answer_complete"] and r["first_assistant_token"] >= 0 and not names_animal.search(r["answer"]):
        seg("ans", r["first_assistant_token"], r["last_assistant_token"] + 1)
        c["close"] = r["last_assistant_temp"]
    for anchor, coord in ANCHORS.items():
        assert coord not in c or c[coord] == r[anchor], (r["uid"], anchor)
    return c


def sort_key(coord):
    seg, off = re.fullmatch(r"([a-z]+)([+-]\d+)?", coord).groups()
    off = int(off or 0)
    return SEG_ORDER.index(seg), off < 0, off           # +0 +1 ... then -K ... -1


cmaps = [coordinates(r) for r in rows]
COORDS = sorted({k for c in cmaps for k in c}, key=sort_key)
T = {c: np.array([m.get(c, -1) for m in cmaps]) for c in COORDS}   # token index per row, -1 = none

tok = AutoTokenizer.from_pretrained(MODEL)
out_dir = Path("results_pretest") / datetime.now().strftime("%m%d_%H%M%S")
out_dir.mkdir(parents=True)
with open(out_dir / "coords.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["coord", "n_rows", "top_token", "top_share", "anchor"])
    inv = {v: k for k, v in ANCHORS.items()}
    for c in COORDS:
        have = np.nonzero(T[c] >= 0)[0]
        (top, cnt), = Counter(rows[i]["input_ids"][T[c][i]] for i in have).most_common(1)
        w.writerow([c, len(have), tok.decode([top]), f"{cnt / len(have):.2f}", inv.get(c, "")])
print(f"{len(COORDS)} coordinates -> {out_dir}")


# ---------- activations ----------
def gather(model, coords):
    """h00..h(L-1) at the given coordinates for every row: {coord: [L, n, D] bf16 on CPU}."""
    L, D = model.config.num_hidden_layers, model.config.hidden_size
    buf = {c: torch.zeros(L, n, D, dtype=torch.bfloat16) for c in coords}
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    by_len = np.argsort([len(r["input_ids"]) for r in rows])
    for s in range(0, n, BATCH):
        b = by_len[s:s + BATCH]
        seqs = [rows[i]["input_ids"] for i in b]
        ids = torch.full((len(b), max(map(len, seqs))), pad)
        mask = torch.zeros_like(ids)
        for j, q in enumerate(seqs):                    # right padding: positions unchanged
            ids[j, :len(q)], mask[j, :len(q)] = torch.tensor(q), 1
        with torch.inference_mode():
            hs = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE),
                       output_hidden_states=True).hidden_states[:L]
        for c in coords:
            t = T[c][b]
            ok = np.nonzero(t >= 0)[0]
            if len(ok):
                bi, ti = torch.as_tensor(ok, device=DEVICE), torch.as_tensor(t[ok], device=DEVICE)
                buf[c][:, torch.as_tensor(b[ok])] = torch.stack([h[bi, ti] for h in hs]).cpu()
    return buf


def reproduce(X):
    """prompt_last is the position Path 1 stored: the states must match its activations."""
    for f in ("h00", "h02", "h21", "h31"):
        ref = torch.as_tensor(data.feature(f, tab_rows[sel])).float()
        mine = X[int(f[1:])].float()
        rel = (mine - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-6)
        print(f"  reproduce Path 1 {f}: median rel. diff {rel.median():.1e}, max {rel.max():.1e}")


# ---------- probes ----------
def fit(Z, y, n_classes):
    """Independent logistic regressions (one per Z[l]), summed losses, one L-BFGS run.
    Same objective per problem as gpu_probe.py: mean CE + ||W||^2 / (2 C n)."""
    L, m, D = Z.shape
    W = torch.zeros(L, D, n_classes, device=DEVICE, requires_grad=True)
    b = torch.zeros(L, 1, n_classes, device=DEVICE, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=1000, history_size=20, tolerance_grad=1e-5,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")
    lam, yy = 1 / (2 * C * m), y.repeat(L)

    def closure():
        opt.zero_grad()
        logits = torch.baddbmm(b, Z, W).reshape(-1, n_classes)
        loss = L * F.cross_entropy(logits, yy) + lam * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach(), opt.state[W]["n_iter"]


def probe(X, coord, writer):
    L = X.shape[0]
    feats = [(f"h{k:02d}", k, None) for k in range(L)] + [(f"d{k:02d}", k, k - 1) for k in range(1, L)]
    have = T[coord] >= 0
    parts = [(name, idx["train"][have[idx["train"]]], idx["test"][have[idx["test"]]]) for name, idx in conds]
    n_match = min(len(tr) for _, tr, _ in parts)        # same train size in every condition
    for name, tr, te in parts:
        if n_match < MIN_TRAIN or len(np.unique(y_np[te])) < 2:
            continue
        tr = np.asarray(data.match_size(tr, y_np, n_match, seed=SEED))
        need = torch.as_tensor(np.concatenate([tr, te]))
        ytr = torch.as_tensor(y_np[tr], device=DEVICE)
        yte = torch.as_tensor(y_np[te], device=DEVICE)
        for s in range(0, len(feats), FEAT_CHUNK):
            chunk = feats[s:s + FEAT_CHUNK]
            Z = torch.stack([X[k][need].to(DEVICE).float() -
                             (X[j][need].to(DEVICE).float() if j is not None else 0)
                             for _, k, j in chunk])
            mu = Z[:, :len(tr)].mean(1, keepdim=True)
            sd = Z[:, :len(tr)].std(1, unbiased=False, keepdim=True)
            sd[sd == 0] = 1
            Z = (Z - mu) / sd
            W, b, it = fit(Z[:, :len(tr)], ytr, len(classes))
            acc = (torch.baddbmm(b, Z[:, len(tr):], W).argmax(-1) == yte).float().mean(1)
            writer.writerows([coord, f, name, len(tr), len(te), f"{a:.4f}", it]
                             for (f, _, _), a in zip(chunk, acc.tolist()))
            del Z


model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEVICE).eval()
res = open(out_dir / "results.csv", "w", newline="")
writer = csv.writer(res)
writer.writerow(["coord", "feature", "condition", "n_train", "n_test", "acc", "n_iter"])
for p in range(0, len(COORDS), POS_PER_PASS):
    coords = COORDS[p:p + POS_PER_PASS]
    t0 = time.time()
    buf = gather(model, coords)
    t1 = time.time()
    if PROMPT_LAST in buf:
        reproduce(buf[PROMPT_LAST])
    for c in coords:
        probe(buf.pop(c), c, writer)
        res.flush()
    print(f"{', '.join(coords):40} forward {t1 - t0:5.0f}s  probes {time.time() - t1:5.0f}s  "
          f"[{min(p + POS_PER_PASS, len(COORDS))}/{len(COORDS)}]", flush=True)
