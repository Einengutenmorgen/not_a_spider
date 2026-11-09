"""Path 2: probe every token from the identifier on, aligned by segment -- for any model in models.py.

    MODEL_KEY=olmo32 RUN_DIR=/tmp/hau/olmo32 ACT_DIR=/GPUdata/hau/acts_seg/olmo32 PLAN_ONLY=1 python3 run_segments.py
    MODEL_KEY=olmo32 RUN_DIR=... ACT_DIR=... ONLY_COORDS=newline,end_of_ident python3 -u run_segments.py   # a smoke test
    MODEL_KEY=olmo32 RUN_DIR=... ACT_DIR=... CUDA_VISIBLE_DEVICES=1 nohup python3 -u run_segments.py > log 2>&1 &

RUN_DIR holds positions.jsonl (index_positions.py) and gets results_segments/. Everything model-specific comes from
models.py: the model id, dtype, forward batch size, coordinates per pass, the assistant-header names, the condition
families and the coordinate prefixes that are left out. The row order is the canonical order of cases.py, so the
splits (data.match_size picks by row position) are the Olmo-7B ones for every model.

Coordinates: every token from the identifier on gets a name. Tokens that every case has are named by their role;
the tokens inside the identifier, the question text after it and the answer are counted from both ends of their
segment.

    identifier        start_of_ident, intermediate_ident_1, _2, ... and ..., _-2, _-1, end_of_ident
    question text     after_ident, intermediate_question_1, ... , _-1, end_of_question
    template          end_user_formatting       the token closing the user turn (<|im_end|>)
                      newline                   \\n between the user and the assistant turn
                      begin_assistant_formatting turn_start token of the assistant turn
                      cfg.header names          Olmo: assistant_role, assistant_header_newline (Path 1's prompt_last)
                                                Qwen, thinking off: assistant_role, assistant_header_newline,
                                                think_open, think_open_newline, think_close, think_close_newline
    answer            begin_of_answer, intermediate_answer_1, ... , _-1, end_of_answer
                      (the whole generated text between assistant header and the closing token)
    template          end_assistant_formatting  closing token (appended, not generated, where the answer hit the budget)

intermediate_X_n is the n-th token after the first one, intermediate_X_-n the n-th before the last one; both only
cover tokens strictly between first and last, so every intermediate token is probed twice. In a one-token segment
the first and the last token are the same token, so the case appears under both names. Not tested: system turn,
user header, question text before the identifier.

One classifier per coordinate, over all cases that have that coordinate. The absolute token index t of every test
case is saved with the predictions, for the join with tokens.parquet.

Per pass of cfg.pos_per_pass coordinates:
  1. extract: one forward pass over the teacher-forced input_ids from positions.jsonl, each case cut after the last
     token the pass needs; $ACT_DIR/<coord>/hLL.npy holds one row per case (native bfloat16 as raw int16 bits),
     rows.npy their indices into the canonical table
  2. probe: h00..h{L-1} and d01..d{L-1} in every condition of cfg.conditions; standardise on train, no PCA, logistic
     regression C=1, L-BFGS as in gpu_probe.py; a condition with < MIN_TRAIN train cases or < 2 animals in test is
     skipped on its own, the others are matched to the smallest train size among themselves
     -> $RUN_DIR/results_segments/results/<coord>.csv and preds/<coord>.npz
  3. delete $ACT_DIR/<coord>; the results are copied to RESULTS_DEST with rclone (RESULTS_DEST="" skips, default)

Parallel: process SHARD of N_SHARDS takes every N_SHARDS-th coordinate. Resumable: rerun the same command; probed
coordinates are skipped, extracted ones are not extracted again.
"""
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import cases
import data
import models

try:                                    # planning and the tests need no GPU stack
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

ANCHORS = ["start_of_ident", "end_of_ident", "end_user_formatting", "newline", "first_assistant_temp",
           "prompt_last", "first_assistant_token", "last_assistant_token", "last_assistant_temp"]
MIN_TRAIN, SEED, C = 200, 0, 1.0


# ---------- cases, splits, coordinates (no torch) ----------
def table(animals, cfg):
    """Canonical rows after the analysis drops, labels, and the conditions to probe."""
    _, tab = data.analysis_rows(cases.canonical_rows(animals))
    uids = np.array([r["uid"] for r in tab])
    classes, y = np.unique([r["animal"] for r in tab], return_inverse=True)
    conds = data.conditions(tab, seed=SEED)
    assert all(len(idx["dev"]) == 0 for _, _, idx in conds), \
        "data.py still holds out dev identifiers: holdout_identifiers(tab, n_test=4, n_dev=0, ...)"
    families = {name.split("/")[0] for name, _, _ in conds}
    assert set(cfg.conditions) <= families, f"unknown condition family in {cfg.conditions}; have {sorted(families)}"
    return uids, classes, y, [c for c in conds if c[0].split("/")[0] in cfg.conditions]


def load_positions(path, uids):
    pos = {}
    for line in open(path):
        r = json.loads(line)
        pos[r["uid"]] = (r["input_ids"], [r[a] for a in ANCHORS])
    ids = [pos[u][0] for u in uids]
    return ids, dict(zip(ANCHORS, np.array([pos[u][1] for u in uids]).T))


def coordinates(A, cfg):
    """Absolute token index per case for every coordinate, -1 where the case has none."""
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
    assert (A["prompt_last"] == fa + len(cfg.header)).all(), f"assistant header is not {len(cfg.header)} tokens"
    T["end_user_formatting"], T["newline"], T["begin_assistant_formatting"] = eu, A["newline"], fa
    for k, (name, _) in enumerate(cfg.header):
        T[name] = fa + 1 + k
    segment("begin_of_answer", "intermediate_answer", "end_of_answer",
            A["first_assistant_token"], A["last_assistant_token"] + 1)
    T["end_assistant_formatting"] = A["last_assistant_temp"]
    return T


def select_coords(T, cfg, only=None):
    cs = [c for c in T if not c.startswith(cfg.skip_coords)]          # str.startswith(()) is False
    if only:
        assert set(only) <= set(cs), f"unknown or skipped coordinates: {sorted(set(only) - set(cs))}"
        cs = [c for c in cs if c in only]
    return cs


def setup(cfg, run_dir, act_dir, data_path="animals.json", only=None, shard=0, n_shards=1, device="cuda"):
    from transformers import AutoConfig
    tcfg = AutoConfig.from_pretrained(cfg.id).get_text_config()      # nested for the Qwen wrapper
    animals = cases.load_animals(data_path)
    uids, classes, y, conds = table(animals, cfg)
    ids, A = load_positions(Path(run_dir) / "positions.jsonl", uids)
    T = coordinates(A, cfg)
    coords = select_coords(T, cfg, only)
    return SimpleNamespace(cfg=cfg, out=Path(run_dir) / "results_segments", act=Path(act_dir), uids=uids,
                           classes=classes, y=y, conds=conds, ids=ids, T=T, coords=coords, device=device,
                           L=tcfg.num_hidden_layers, D=tcfg.hidden_size, shard=shard, n_shards=n_shards,
                           n_rows={c: int((T[c] >= 0).sum()) for c in coords})


# ---------- extract ----------
def act_dir(S, c):
    return S.act / c


def probed(S, c):
    return (S.out / "results" / f"{c}.csv").exists()


def extracted(S, c):
    return (act_dir(S, c) / "complete").exists()


def sanity(model, S):
    """One forward pass: the model must return L+1 hidden states (h[L] = the normed last one, never used)."""
    x = torch.tensor([S.ids[0][:S.T["end_of_ident"][0] + 1]], device=S.device)
    with torch.inference_mode():
        hs = model(input_ids=x, attention_mask=torch.ones_like(x), output_hidden_states=True).hidden_states
    norms = [f"h{k:02d}={hs[k][0, -1].float().norm():.0f}" for k in sorted({0, S.L // 2, S.L - 1, S.L} & set(range(len(hs))))]
    print(f"{type(model).__name__}: {len(hs)} hidden states, dtype {hs[0].dtype}, last-token norms {' '.join(norms)}", flush=True)
    assert len(hs) == S.L + 1, f"expected {S.L + 1} hidden states, got {len(hs)}"


def extract(model, cs, S):
    gib = sum(S.n_rows[c] for c in cs) * S.L * S.D * 2 / 2**30
    # the pool for activations is the free space plus what activations hold right now (other
    # processes' current passes); it must fit n_shards passes of this size at once
    free = shutil.disk_usage(S.act).free / 2**30
    held = sum(f.stat().st_blocks * 512 for f in S.act.rglob("*.npy")) / 2**30
    assert gib * S.n_shards + 2 < free + held, \
        f"{cs}: {S.n_shards} process(es) need {gib * S.n_shards:.0f} GiB, {free + held:.0f} GiB available on {S.act}"
    out, slot = {}, {}
    for c in cs:
        act_dir(S, c).mkdir(exist_ok=True)
        rows_c = np.nonzero(S.T[c] >= 0)[0]
        np.save(act_dir(S, c) / "rows.npy", rows_c)
        slot[c] = np.full(len(S.uids), -1)
        slot[c][rows_c] = np.arange(len(rows_c))
        out[c] = [np.lib.format.open_memmap(act_dir(S, c) / f"h{k:02d}.npy", mode="w+", dtype=np.int16,
                                            shape=(len(rows_c), S.D)) for k in range(S.L)]
    last = np.max([S.T[c] for c in cs], axis=0)                      # last token a case needs
    rows = np.nonzero(last >= 0)[0]
    rows = rows[np.argsort(last[rows], kind="stable")]
    pad = getattr(model.config.get_text_config(), "pad_token_id", None) or 0
    for s in range(0, len(rows), S.cfg.fwd_batch):
        b = rows[s:s + S.cfg.fwd_batch]
        seqs = [S.ids[i][:last[i] + 1] for i in b]
        x = torch.full((len(b), max(map(len, seqs))), pad)
        mask = torch.zeros_like(x)
        for j, q in enumerate(seqs):                                 # right padding
            x[j, :len(q)], mask[j, :len(q)] = torch.tensor(q), 1
        with torch.inference_mode():
            hs = model(input_ids=x.to(S.device), attention_mask=mask.to(S.device),
                       output_hidden_states=True).hidden_states[:S.L]
        for c in cs:
            ok = np.nonzero(S.T[c][b] >= 0)[0]
            if len(ok):
                bi = torch.as_tensor(ok, device=S.device)
                ti = torch.as_tensor(S.T[c][b[ok]], device=S.device)
                st = torch.stack([h[bi, ti] for h in hs])
                assert st.dtype == torch.bfloat16 and torch.isfinite(st).all()
                st = st.view(torch.int16).cpu().numpy()                     # raw bfloat16 bits
                for k in range(S.L):
                    out[c][k][slot[c][b[ok]]] = st[k]
    for c in cs:
        for m in out[c]:
            m.flush()
        (act_dir(S, c) / "complete").touch()


# ---------- probe ----------
def fit(Z, yy, n_classes, device):
    """Independent logistic regressions, one per Z[f], summed losses in one L-BFGS run.
    Same objective per problem as gpu_probe.py: mean CE + ||W||^2 / (2 C n)."""
    nf, m, d = Z.shape
    W = torch.zeros(nf, d, n_classes, device=device, requires_grad=True)
    b = torch.zeros(nf, 1, n_classes, device=device, requires_grad=True)
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


def probe(c, S):
    dev, L = S.device, S.L
    rows = np.load(act_dir(S, c) / "rows.npy")
    present = np.zeros(len(S.uids), bool)
    present[rows] = True
    slot = np.full(len(S.uids), -1)
    slot[rows] = np.arange(len(rows))
    parts = [(name, idx["train"][present[idx["train"]]], idx["test"][present[idx["test"]]])
             for name, _, idx in S.conds]
    valid = {name for name, tr, te in parts if len(tr) >= MIN_TRAIN and len(np.unique(S.y[te])) >= 2}
    n_match = min((len(tr) for name, tr, _ in parts if name in valid), default=0)
    feats = [(f"h{k:02d}", k, None) for k in range(L)] + [(f"d{k:02d}", k, k - 1) for k in range(1, L)]
    res, preds = [], {}
    for s in range(0, len(feats), S.cfg.feat_chunk):
        chunk = feats[s:s + S.cfg.feat_chunk]
        H = {k: torch.from_numpy(np.load(act_dir(S, c) / f"h{k:02d}.npy")).to(dev).view(torch.bfloat16)
             for k in {k for _, k, _ in chunk} | {j for _, _, j in chunk if j is not None}}
        X = torch.stack([H[k].float() - (H[j].float() if j is not None else 0) for _, k, j in chunk])
        del H
        for name, tr, te in parts:
            if name not in valid:
                res += [[c, f, name, len(tr), len(te), "", "skipped"] for f, _, _ in chunk]
                continue
            tr = np.asarray(data.match_size(tr, S.y, n_match, seed=SEED))
            Z = X[:, torch.as_tensor(slot[np.concatenate([tr, te])], device=dev)]
            mu = Z[:, :len(tr)].mean(1, keepdim=True)
            sd = Z[:, :len(tr)].std(1, unbiased=False, keepdim=True)
            sd[sd == 0] = 1
            Z.sub_(mu).div_(sd)
            W, b, it = fit(Z[:, :len(tr)], torch.as_tensor(S.y[tr], device=dev), len(S.classes), dev)
            pred = torch.baddbmm(b, Z[:, len(tr):], W).argmax(-1)                     # [features, test]
            acc = (pred == torch.as_tensor(S.y[te], device=dev)).float().mean(1)
            res += [[c, f, name, len(tr), len(te), f"{a:.4f}", it] for (f, _, _), a in zip(chunk, acc.tolist())]
            preds.setdefault(name, [te]).append(pred.to(torch.int8).cpu().numpy())
            del Z
        del X
    torch.cuda.empty_cache()
    key = lambda n: n.replace("/", "_")
    np.savez_compressed(S.out / "preds" / f"{c}.npz", classes=S.classes, features=[f for f, _, _ in feats],
                        **{f"{key(n)}_rows": p[0] for n, p in preds.items()},             # into uids.npy
                        **{f"{key(n)}_t": S.T[c][p[0]] for n, p in preds.items()},          # absolute index
                        **{f"{key(n)}_pred": np.concatenate(p[1:]) for n, p in preds.items()})
    tmp = S.out / "results" / f"{c}.tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["coord", "feature", "condition", "n_train", "n_test", "acc", "n_iter"])
        w.writerows(res)
    tmp.replace(S.out / "results" / f"{c}.csv")                     # written last: marks c as probed


def backup(S, dest):
    if dest:
        r = subprocess.run(["rclone", "copy", str(S.out), dest], capture_output=True, text=True)
        if r.returncode:
            print(f"results backup failed (kept locally): {r.stderr.strip()[-300:]}", flush=True)


# ---------- run ----------
def main():
    key = os.environ.get("MODEL_KEY")
    if key not in models.MODELS:
        raise SystemExit(f"set MODEL_KEY to one of {list(models.MODELS)}")
    cfg = models.MODELS[key]
    run_dir = os.environ.get("RUN_DIR", f"runs/{key}")
    only = [c for c in os.environ.get("ONLY_COORDS", "").split(",") if c]
    S = setup(cfg, run_dir, os.environ.get("ACT_DIR", f"acts_seg/{key}"), only=only,
              shard=int(os.environ.get("SHARD", 0)), n_shards=int(os.environ.get("N_SHARDS", 1)),
              device=os.environ.get("DEVICE", "cuda"))
    (S.out / "results").mkdir(parents=True, exist_ok=True)
    (S.out / "preds").mkdir(exist_ok=True)
    S.act.mkdir(parents=True, exist_ok=True)
    if S.shard == 0:                                                 # shared files: one writer
        np.save(S.out / "uids.npy", S.uids)                         # preds index into this
        with open(S.out / "plan.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["coord", "n_rows", "gib"])
            w.writerows([c, S.n_rows[c], f"{S.n_rows[c] * S.L * S.D * 2 / 2**30:.1f}"] for c in S.coords)
    print(f"{cfg.id}: {len(S.uids)} cases, conditions {[c for c, _, _ in S.conds]} | {len(S.coords)} coordinates, "
          f"{sum(S.n_rows.values()) * S.L * S.D * 2 / 2**40:.1f} TiB to extract in total | L={S.L} D={S.D}", flush=True)
    if os.environ.get("PLAN_ONLY"):
        raise SystemExit
    mine = S.coords[S.shard::S.n_shards]

    for c in mine:                                  # left over from an interrupted run
        if probed(S, c) and act_dir(S, c).exists():
            shutil.rmtree(act_dir(S, c))
    todo = [c for c in mine if not probed(S, c)]
    print(f"shard {S.shard}/{S.n_shards}: {len(todo)} coordinates to do", flush=True)
    if not todo:
        return
    from transformers import AutoModel
    if S.device == "cuda":                          # weights + the hidden states of one batch must fit
        hs_gib = cfg.fwd_batch * 160 * S.D * (S.L + 1) * 2 / 2**30
        models.check_vram(cfg, headroom_gib=1.5 * hs_gib + 4)
    model = AutoModel.from_pretrained(cfg.id, dtype=getattr(torch, cfg.dtype)).eval()
    checked = False
    for p in range(0, len(todo), cfg.pos_per_pass):
        cs = todo[p:p + cfg.pos_per_pass]
        t0 = time.time()
        if missing := [c for c in cs if not extracted(S, c)]:
            model.to(S.device)
            if not checked:
                sanity(model, S)
                checked = True
            extract(model, missing, S)
            model.to("cpu")                                          # free the GPU for probing
        t1 = time.time()
        for c in cs:
            probe(c, S)
            shutil.rmtree(act_dir(S, c))
        backup(S, os.environ.get("RESULTS_DEST", ""))
        print(f"{', '.join(cs):20}  extract {t1 - t0:5.0f}s  probe {time.time() - t1:5.0f}s  "
              f"[{p + len(cs)}/{len(todo)}]", flush=True)


if __name__ == "__main__":
    main()