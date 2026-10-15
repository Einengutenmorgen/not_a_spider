"""Path 2: probe every token from the identifier on, aligned by segment.

    PLAN_ONLY=1 python3 run_segments.py                                    # plan.csv, then stop

    GPUS=(2 3)
    for i in "${!GPUS[@]}"; do
      CUDA_VISIBLE_DEVICES=${GPUS[$i]} SHARD=$i N_SHARDS=${#GPUS[@]} HF_HOME=/media/data/hau/hf \
        nohup python3 -u run_segments.py > logs/segments_s${i}_$(date +%m%d_%H%M).log 2>&1 &
    done

Coordinates: every token from the identifier on gets a name. Tokens that every case has are
named by their role; the tokens inside the identifier, the question text after it and the
answer are counted from both ends of their segment.

    identifier        start_of_ident, intermediate_ident_1, _2, ... and ..., _-2, _-1, end_of_ident
    question text     after_ident, intermediate_question_1, ... , _-1, end_of_question
    template          end_user_formatting       <|im_end|> closing the user turn
                      newline                   \\n between the user and the assistant turn
                      begin_assistant_formatting <|im_start|> of the assistant turn
                      assistant_role            the "assistant" token
                      assistant_header_newline  \\n after "assistant" (Path 1's prompt_last)
    answer            begin_of_answer, intermediate_answer_1, ... , _-1, end_of_answer
                      (the whole generated text between assistant header and <|im_end|>)
    template          end_assistant_formatting  <|im_end|> closing the answer (appended, not
                                                generated, where the answer hit the token budget)

intermediate_X_n is the n-th token after the first one, intermediate_X_-n the n-th before the
last one; both only cover tokens strictly between first and last, so every intermediate token
is probed twice. In a one-token segment the first and the last token are the same token, so the
case appears under both names. Not tested: system turn, user header, question text before the
identifier.

One classifier per coordinate, over all cases that have that coordinate. The absolute token
index t of every test case is saved with the predictions, for the join with tokens.parquet.

Per pass of POS_PER_PASS coordinates:
  1. extract: one forward pass over the teacher-forced input_ids from positions.jsonl, each
     case cut after the last token the pass needs; $ACT_DIR/<coord>/hLL.npy holds one row per
     case (native bfloat16 as raw int16 bits), rows.npy their indices into data.analysis_rows
  2. probe: h00..h31 and d01..d31 in every condition of data.conditions; standardise on train,
     no PCA, logistic regression C=1, L-BFGS as in gpu_probe.py; a condition with < MIN_TRAIN
     train cases or < 2 animals in test is skipped on its own, the others are matched to the
     smallest train size among themselves
     -> $OUT/results/<coord>.csv and $OUT/preds/<coord>.npz
  3. delete $ACT_DIR/<coord>; $OUT is copied to RESULTS_DEST with rclone (RESULTS_DEST="" skips)

Parallel: process SHARD of N_SHARDS takes every N_SHARDS-th coordinate. Resumable: rerun the
same command; probed coordinates are skipped, extracted ones are not extracted again.
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
ACT_DIR = Path(os.environ.get("ACT_DIR", "/media/data/hau/acts_seg"))
OUT = Path(os.environ.get("OUT", "results_segments"))
RESULTS_DEST = os.environ.get("RESULTS_DEST", "NotaSpider:not_a_spider/segments/results")
DEVICE = os.environ.get("DEVICE", "cuda")
POS_PER_PASS = int(os.environ.get("POS_PER_PASS", 2))
FEAT_CHUNK = int(os.environ.get("FEAT_CHUNK", 2))    # features fitted together; GPU ~16 GB at 2
SHARD, N_SHARDS = int(os.environ.get("SHARD", 0)), int(os.environ.get("N_SHARDS", 1))
BATCH, MIN_TRAIN, SEED, C = 64, 200, 0, 1.0

# ---------- cases and splits ----------
_, tab = data.analysis_rows(data.build_table())
uids = np.array([r["uid"] for r in tab])
classes, y = np.unique([r["animal"] for r in tab], return_inverse=True)
conds = data.conditions(tab, seed=SEED)
assert all(len(idx["dev"]) == 0 for _, _, idx in conds), \
    "data.py still holds out dev identifiers: holdout_identifiers(tab, n_test=4, n_dev=0, ...)"

ANCHORS = ["start_of_ident", "end_of_ident", "end_user_formatting", "newline", "first_assistant_temp",
           "prompt_last", "first_assistant_token", "last_assistant_token", "last_assistant_temp"]
pos = {}
for line in open("positions.jsonl"):
    r = json.loads(line)
    pos[r["uid"]] = (r["input_ids"], [r[a] for a in ANCHORS])
ids = [pos[u][0] for u in uids]
A = dict(zip(ANCHORS, np.array([pos[u][1] for u in uids]).T))
del pos

# ---------- coordinates: absolute token index per case, -1 where the case has none ----------
T = {}


def segment(first, middle, last, a, b):
    """Tokens a..b-1 of every case: first and last by name, the ones in between counted from
    the start (middle_1, middle_2, ...) and from the end (..., middle_-2, middle_-1)."""
    n = np.where(a >= 0, b - a, 0)
    T[first] = np.where(n > 0, a, -1)
    for k in range(1, n.max() - 1):
        T[f"{middle}_{k}"] = np.where(n - 1 > k, a + k, -1)
    for k in range(n.max() - 2, 0, -1):
        T[f"{middle}_-{k}"] = np.where(n - 1 > k, b - 1 - k, -1)
    T[last] = np.where(n > 0, b - 1, -1)


segment("start_of_ident", "intermediate_ident", "end_of_ident", A["start_of_ident"], A["end_of_ident"] + 1)
segment("after_ident", "intermediate_question", "end_of_question", A["end_of_ident"] + 1, A["end_user_formatting"])
eu, fa = A["end_user_formatting"], A["first_assistant_temp"]
assert (A["newline"] == eu + 1).all() and (fa == eu + 2).all(), "unexpected template layout"
role = A["prompt_last"] - fa - 1                                     # tokens in between: 1 for Olmo
assert (role == role[0]).all(), "assistant header length varies"
T["end_user_formatting"], T["newline"], T["begin_assistant_formatting"] = eu, A["newline"], fa
for k in range(role[0]):
    T["assistant_role" if role[0] == 1 else f"assistant_role_{k}"] = fa + 1 + k
T["assistant_header_newline"] = A["prompt_last"]
segment("begin_of_answer", "intermediate_answer", "end_of_answer",
        A["first_assistant_token"], A["last_assistant_token"] + 1)
T["end_assistant_formatting"] = A["last_assistant_temp"]
COORDS = list(T)                                                     # string order

cfg = AutoConfig.from_pretrained(MODEL)
L, D = cfg.num_hidden_layers, cfg.hidden_size                       # h00..h31 as in Path 1
n_rows = {c: int((T[c] >= 0).sum()) for c in COORDS}
(OUT / "results").mkdir(parents=True, exist_ok=True)
(OUT / "preds").mkdir(exist_ok=True)
ACT_DIR.mkdir(parents=True, exist_ok=True)
if SHARD == 0:                                                       # shared files: one writer
    np.save(OUT / "uids.npy", uids)                                 # preds index into this
    with open(OUT / "plan.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["coord", "n_rows", "gib"])
        w.writerows([c, n_rows[c], f"{n_rows[c] * L * D * 2 / 2**30:.1f}"] for c in COORDS)
print(f"{len(uids)} cases, conditions {[c for c, _, _ in conds]} | {len(COORDS)} coordinates, "
      f"{sum(n_rows.values()) * L * D * 2 / 2**40:.1f} TiB to extract in total", flush=True)
if os.environ.get("PLAN_ONLY"):
    raise SystemExit
mine = COORDS[SHARD::N_SHARDS]


def act_dir(c):
    return ACT_DIR / c


def probed(c):
    return (OUT / "results" / f"{c}.csv").exists()


def extracted(c):
    return (act_dir(c) / "complete").exists()


# ---------- extract ----------
def extract(model, cs):
    gib = sum(n_rows[c] for c in cs) * L * D * 2 / 2**30
    # the pool for activations is the free space plus what activations hold right now (other
    # processes' current passes); it must fit N_SHARDS passes of this size at once
    free = shutil.disk_usage(ACT_DIR).free / 2**30
    held = sum(f.stat().st_blocks * 512 for f in ACT_DIR.rglob("*.npy")) / 2**30
    assert gib * N_SHARDS + 2 < free + held, \
        f"{cs}: {N_SHARDS} process(es) need {gib * N_SHARDS:.0f} GiB, {free + held:.0f} GiB available on {ACT_DIR}"
    out, slot = {}, {}
    for c in cs:
        act_dir(c).mkdir(exist_ok=True)
        rows_c = np.nonzero(T[c] >= 0)[0]
        np.save(act_dir(c) / "rows.npy", rows_c)
        slot[c] = np.full(len(uids), -1)
        slot[c][rows_c] = np.arange(len(rows_c))
        out[c] = [np.lib.format.open_memmap(act_dir(c) / f"h{k:02d}.npy", mode="w+", dtype=np.int16,
                                            shape=(len(rows_c), D)) for k in range(L)]
    last = np.max([T[c] for c in cs], axis=0)                        # last token a case needs
    rows = np.nonzero(last >= 0)[0]
    rows = rows[np.argsort(last[rows], kind="stable")]
    pad = model.config.pad_token_id or 0
    for s in range(0, len(rows), BATCH):
        b = rows[s:s + BATCH]
        seqs = [ids[i][:last[i] + 1] for i in b]
        x = torch.full((len(b), max(map(len, seqs))), pad)
        mask = torch.zeros_like(x)
        for j, q in enumerate(seqs):                                 # right padding
            x[j, :len(q)], mask[j, :len(q)] = torch.tensor(q), 1
        with torch.inference_mode():
            hs = model(input_ids=x.to(DEVICE), attention_mask=mask.to(DEVICE),
                       output_hidden_states=True).hidden_states[:L]
        for c in cs:
            ok = np.nonzero(T[c][b] >= 0)[0]
            if len(ok):
                bi = torch.as_tensor(ok, device=DEVICE)
                ti = torch.as_tensor(T[c][b[ok]], device=DEVICE)
                st = torch.stack([h[bi, ti] for h in hs])
                assert st.dtype == torch.bfloat16 and torch.isfinite(st).all()
                st = st.view(torch.int16).cpu().numpy()                     # raw bfloat16 bits
                for k in range(L):
                    out[c][k][slot[c][b[ok]]] = st[k]
    for c in cs:
        for m in out[c]:
            m.flush()
        (act_dir(c) / "complete").touch()


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


def probe(c):
    rows = np.load(act_dir(c) / "rows.npy")
    present = np.zeros(len(uids), bool)
    present[rows] = True
    slot = np.full(len(uids), -1)
    slot[rows] = np.arange(len(rows))
    parts = [(name, idx["train"][present[idx["train"]]], idx["test"][present[idx["test"]]])
             for name, _, idx in conds]
    valid = {name for name, tr, te in parts if len(tr) >= MIN_TRAIN and len(np.unique(y[te])) >= 2}
    n_match = min((len(tr) for name, tr, _ in parts if name in valid), default=0)
    feats = [(f"h{k:02d}", k, None) for k in range(L)] + [(f"d{k:02d}", k, k - 1) for k in range(1, L)]
    res, preds = [], {}
    for s in range(0, len(feats), FEAT_CHUNK):
        chunk = feats[s:s + FEAT_CHUNK]
        H = {k: torch.from_numpy(np.load(act_dir(c) / f"h{k:02d}.npy")).to(DEVICE).view(torch.bfloat16)
             for k in {k for _, k, _ in chunk} | {j for _, _, j in chunk if j is not None}}
        X = torch.stack([H[k].float() - (H[j].float() if j is not None else 0) for _, k, j in chunk])
        del H
        for name, tr, te in parts:
            if name not in valid:
                res += [[c, f, name, len(tr), len(te), "", "skipped"] for f, _, _ in chunk]
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
            res += [[c, f, name, len(tr), len(te), f"{a:.4f}", it] for (f, _, _), a in zip(chunk, acc.tolist())]
            preds.setdefault(name, [te]).append(pred.to(torch.int8).cpu().numpy())
            del Z
        del X
    torch.cuda.empty_cache()
    key = lambda n: n.replace("/", "_")
    np.savez_compressed(OUT / "preds" / f"{c}.npz", classes=classes, features=[f for f, _, _ in feats],
                        **{f"{key(n)}_rows": p[0] for n, p in preds.items()},             # into uids.npy
                        **{f"{key(n)}_t": T[c][p[0]] for n, p in preds.items()},          # absolute index
                        **{f"{key(n)}_pred": np.concatenate(p[1:]) for n, p in preds.items()})
    tmp = OUT / "results" / f"{c}.tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["coord", "feature", "condition", "n_train", "n_test", "acc", "n_iter"])
        w.writerows(res)
    tmp.replace(OUT / "results" / f"{c}.csv")                       # written last: marks c as probed


def backup():
    if RESULTS_DEST:
        r = subprocess.run(["rclone", "copy", str(OUT), RESULTS_DEST], capture_output=True, text=True)
        if r.returncode:
            print(f"results backup failed (kept locally): {r.stderr.strip()[-300:]}", flush=True)


# ---------- run ----------
for c in mine:                                      # left over from an interrupted run
    if probed(c) and act_dir(c).exists():
        shutil.rmtree(act_dir(c))
model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
todo = [c for c in mine if not probed(c)]
print(f"shard {SHARD}/{N_SHARDS}: {len(todo)} coordinates to do", flush=True)
for p in range(0, len(todo), POS_PER_PASS):
    cs = todo[p:p + POS_PER_PASS]
    t0 = time.time()
    if missing := [c for c in cs if not extracted(c)]:
        model.to(DEVICE)
        extract(model, missing)
        model.to("cpu")                                              # free the GPU for probing
    t1 = time.time()
    for c in cs:
        probe(c)
        shutil.rmtree(act_dir(c))
    backup()
    print(f"{', '.join(cs):20}  extract {t1 - t0:5.0f}s  probe {time.time() - t1:5.0f}s  "
          f"[{p + len(cs)}/{len(todo)}]", flush=True)