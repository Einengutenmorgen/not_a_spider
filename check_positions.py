"""Manual safety checks for stream_probe.py on 5 examples. Prints everything.

    python check_positions.py              # 5 curated rows
    python check_positions.py --seed 3     # 5 random rows instead

Uses the functions of stream_probe.py itself, so what you see is what the run does.
Needs a GPU for the real model (MODEL env var picks another model). Nothing is written.

Sections
  1  token table per row: index, coordinate, token, markers  -> read the markers by eye
  2  generation: batched (as in prepare) vs one row at a time
  3  every probed position: which token each row feeds there
  4  collect(): stored states vs a plain unbatched forward, h[0] identity, sys_00 spread
  5  h vs d: d[k] == h[k] - h[k-1], norms per layer at assistant_tmp_end
  6  probe mechanics: fit() vs the sklearn pipeline on these 5 rows
  7  splits on the full table (no model): condition sizes, a partial-coverage coordinate
"""
import argparse
from collections import Counter
import sys
from pathlib import Path
import numpy as np
import torch

import data
import stream_probe as sp

from contextlib import redirect_stdout #to save std out to file write 

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

def title(t):
    print(f"\n{'=' * 100}\n{t}\n{'=' * 100}")


def ok(flag, msg):
    print(f"  [{'OK' if flag else 'FAIL'}] {msg}")
    return flag


def pick(rows, seed):
    if seed is not None:
        rng = np.random.default_rng(seed)
        return [(f"random {i}", rows[i]) for i in rng.choice(len(rows), 5, replace=False)]
    first = lambda r: next(c for c in r["coords"] if not c.startswith("sys"))
    rules = [
        ("example from chat (panda, limbs)",
         lambda: next((r for r in rows if "diplomatic gesture" in r["question"]
                       and r["slot"] == "number_of_limbs"), rows[0])),
        ("frame starts with the identifier (no pre segment)",
         lambda: next(r for r in rows if first(r).startswith("id"))),
        ("longest identifier", lambda: max(rows, key=lambda r: sum(
            c.startswith("id") for c in r["coords"]))),
        ("numeric slot (weight)", lambda: next(r for r in rows if r["slot"] == "weight")),
        ("shortest prompt", lambda: min(rows, key=lambda r: r["n_tokens"])),
    ]
    return [(label, f()) for label, f in rules]


def token_table(tok, r):
    ids = r["input_ids"] + r.get("gen_ids", [])
    at = Counter()
    marks = {}
    for m in sp.MARKERS:
        if m in r:
            marks.setdefault(r[m], []).append(m)
    print(f"  {'idx':>4}  {'coord':8}  {'id':>7}  {'token':24}  markers")
    for i, (c, t) in enumerate(zip(r["coords"], ids)):
        at[c.split("_")[0]] += 1
        print(f"  {i:4d}  {c:8}  {t:7d}  {tok.decode([t])!r:24}  {' '.join(marks.get(i, []))}")
    print(f"  segment lengths: {dict(at)}")


def main(seed):
    title(f"0  model {sp.MODEL} on {sp.DEVICE}")
    tok, model = sp.load()
    L = model.config.num_hidden_layers
    print(f"  layers {L} -> states h00..h{L - 1:02d}, deltas d01..d{L - 1:02d}; "
          f"hidden {model.config.hidden_size}; dtype {model.dtype}")
    print(f"  FWD_BATCH {sp.FWD_BATCH}, GEN_BATCH {sp.GEN_BATCH}, MAX_NEW_TOKENS {sp.MAX_NEW_TOKENS}")

    all_rows = sp.build_rows(tok)
    print(f"  built {len(all_rows)} rows from {sp.DATA}")
    chosen = pick(all_rows, seed)
    rows = [r for _, r in chosen]

    # ---------------------------------------------------------------------------------
    title("1  prompt token tables (check every marker by eye)")
    for (label, r) in chosen:
        print(f"\n--- {label} | {r['uid']}")
        print(f"  question: {r['question']}")
        print(f"  templated: {tok.decode(r['input_ids'])!r}")
        token_table(tok, r)
        ok(len(set(r["coords"])) == len(r["coords"]) == r["n_tokens"], "coordinates unique, one per token")
        ok(r["coords"][r["ident_last"]].startswith("id")
           and not r["coords"][r["ident_last"] + 1].startswith("id"), "ident_last is the last id token")
        ok(r["end_user"] + 1 == r["assistant_tmp_begin"], "assistant_tmp_begin directly follows end_user")
        ok(r["assistant_tmp_end"] == r["n_tokens"] - 1, "assistant_tmp_end is the last prompt token (= v1 prompt_last)")
    for seg in ("sys", "tmpl"):
        seqs = {tuple(t for t, c in zip(r["input_ids"], r["coords"]) if c.startswith(seg)) for r in rows}
        ok(len(seqs) == 1, f"{seg} tokens identical across the 5 rows")

    # ---------------------------------------------------------------------------------
    title("2  generation: batched (prepare) vs one at a time")
    sp.generate(model, tok, rows)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    for (label, r) in chosen:
        with torch.inference_mode():
            single = model.generate(torch.tensor([r["input_ids"]], device=sp.DEVICE),
                                    max_new_tokens=sp.MAX_NEW_TOKENS, do_sample=False,
                                    pad_token_id=pad)[0, r["n_tokens"]:].tolist()
        single = single[:len(r["gen_ids"])]
        split = next((t for t, (a_, b_) in enumerate(zip(single, r["gen_ids"])) if a_ != b_), None)
        print(f"\n--- {label}\n  answer {r['answer']!r}  correct={r['correct']}  "
              f"capped={r['capped']}  gen_ids {r['gen_ids']}")
        if split is None:
            ok(True, "unbatched generation gives the same ids")
        else:
            # near-tied greedy steps flip under bf16 when the batch changes; the stored ids are
            # what gets teacher-forced, so probing stays consistent -- but answers depend on GEN_BATCH
            print(f"  [WARN] unbatched generation diverges at answer token {split} "
                  f"({tok.decode(r['gen_ids'][split:split + 1])!r} vs {tok.decode(single[split:split + 1])!r}); "
                  f"identical before: {tok.decode(r['gen_ids'][:split])!r}")
        print("  answer part of the token table:")
        n = r["n_tokens"]
        for i in range(n - 2, n + len(r["gen_ids"])):
            ms = [m for m in sp.MARKERS if r[m] == i]
            print(f"  {i:4d}  {r['coords'][i]:8}  {tok.decode((r['input_ids'] + r['gen_ids'])[i:i + 1])!r:24}  {' '.join(ms)}")
        ends = r["gen_ids"][-1] in eos
        ok(r["last"] == n + len(r["gen_ids"]) - 1, "last is the final token")
        ok(r["answer_last"] == (r["last"] - 1 if ends and len(r["gen_ids"]) > 1 else r["last"]),
           "answer_last precedes the end token" if ends else "answer_last == last")
        if not ends:
            print(f"  [WARN] no end token within {sp.MAX_NEW_TOKENS} tokens: the answer may be cut off, "
                  "answer_last/last sit on the cap, and 'correct' may be wrong")

    # ---------------------------------------------------------------------------------
    title("3  every probed position -> token per row")
    names, aliases = sp.positions(rows)
    where = sp.locate(rows, names)
    print(f"  {len(names)} positions on these 5 rows (the full run has more: longest segments)")
    print(f"  fixed markers (labels): {aliases}")
    print(f"  moving markers (own positions): {[n for n in names if n in sp.MARKERS]}\n")
    print(f"  {'position':14} {'markers':22} " + " ".join(f"{'row' + str(j):18}" for j in range(5)))
    for j, p in enumerate(names):
        cells = []
        for r, i in zip(rows, where[:, j]):
            cells.append(f"{i:3d} {tok.decode((r['input_ids'] + r['gen_ids'])[i:i + 1])!r:.13}" if i >= 0 else "  -")
        print(f"  {p:14} {' '.join(aliases.get(p, [])):22} " + " ".join(f"{c:18}" for c in cells))
    covered = all(set(r["coords"]) <= set(names) for r in rows)
    ok(covered, "every token of every row maps to a probed coordinate")

    # ---------------------------------------------------------------------------------
    title("4  collect(): stored states vs a plain unbatched forward")
    A = sp.collect(model, rows, where, L)
    print(f"  A shape {A.shape} = (positions, states, rows, hidden), dtype {A.dtype}")
    # (a) alignment: stored vs a plain one-row forward. bf16 differs between batch size 32 and 1,
    #     so this is not zero; what matters is that it is far below a one-token shift.
    err, nbr, worst = [], [], []
    for k, r in enumerate(rows):
        seq = r["input_ids"] + r["gen_ids"]
        with torch.inference_mode():
            hs = model(torch.tensor([seq], device=sp.DEVICE), output_hidden_states=True).hidden_states
        ref = torch.stack([h[0] for h in hs[:L]]).float().cpu().numpy()     # (L, T, d)
        e, n, at = [], [], []
        for j in range(len(names)):
            i = where[k, j]
            if i >= 1:
                a_ = A[j, :, k].astype(np.float32)
                size = np.linalg.norm(ref[:, i], axis=-1) + 1e-6
                e.append(np.linalg.norm(a_ - ref[:, i], axis=-1) / size)
                n.append(np.linalg.norm(a_ - ref[:, i - 1], axis=-1) / size)
                at.append(names[j])
        e = np.array(e)                                            # (positions, states)
        err.append(e)
        nbr.append(np.median(n, 0))
        jj, ss = np.unravel_index(e.argmax(), e.shape)
        worst.append((e.max(), k, at[jj], ss))
        print(f"  row{k} relative error: median {np.median(e):.2e}  worst {e.max():.2e} "
              f"at {at[jj]} h{ss:02d}")
    med = np.median(np.concatenate(err), 0)
    mx = np.max([e.max(0) for e in err], 0)
    nbr = np.min(nbr, 0)
    print(f"\n  per state: median / worst relative error vs median distance to the neighbouring token")
    for s_ in range(L):
        print(f"    h{s_:02d}  median {med[s_]:.2e}  worst {mx[s_]:.2e}   neighbour {nbr[s_]:.2e}"
              f"   worst ratio {nbr[s_] / max(mx[s_], 1e-12):8.1f}")
    ok((nbr[1:] > 10 * mx[1:]).all(),
       "aligned: every stored state is >10x closer to its own token than to the neighbour")

    # (b) batch invariance: the same rows in a different order and batch slots must give the
    #     same states. This is the property the fixed batch shape exists for.
    B = sp.collect(model, rows, where, L, seed=sp.SEED + 1)
    diff = np.abs(A.astype(np.float32) - B.astype(np.float32)).max(axis=(0, 2, 3))
    print(f"  max |order A - order B| per state: {np.array2string(diff, precision=2, max_line_width=160)}")
    if not ok(diff.max() == 0, "states independent of batch order and slot (bitwise)"):
        rel = diff.max() / np.abs(A.astype(np.float32)).max()
        print(f"      not bitwise; relative {rel:.1e}. Small values are kernel noise, but noise that "
              "depends on batch composition can carry class information at identical-input positions.")
    del B
    shared, bad = 0, []
    for j, p in enumerate(names):
        avail = where[:, j] >= 0
        toks = {(r["input_ids"] + r["gen_ids"])[i] for r, i in zip(rows, where[:, j]) if i >= 0}
        if len(toks) == 1 and avail.sum() > 1:
            shared += 1
            if np.ptp(A[j, 0, avail], axis=0).max() != 0:
                bad.append(p)
    ok(not bad, f"h[0] identical at all {shared} positions where the rows share a token {bad or ''}")
    j0 = names.index("sys_00")
    spread = np.ptp(A[j0].astype(np.float32), axis=1).max(-1)
    print(f"  sys_00 spread across rows per state: {np.array2string(spread, precision=4, max_line_width=160)}")
    ok(spread.max() == 0, "sys_00 identical in every row at every state (fixed batch shape works)")

    # ---------------------------------------------------------------------------------
    title("5  hidden state vs residual delta at assistant_tmp_end")
    jp = names.index(next(p for p, m in aliases.items() if "assistant_tmp_end" in m))
    for k in (1, L // 2, L - 1):
        d = sp.to_gpu(A[jp], "d", k)
        ok(torch.equal(d, sp.to_gpu(A[jp], "h", k) - sp.to_gpu(A[jp], "h", k - 1)),
           f"d{k:02d} == h{k:02d} - h{k - 1:02d}")
    print(f"\n  {'layer':>5} {'|h| mean':>10} {'|d| mean':>10} {'|d|/|h|':>8} {'cos(d, h[k-1])':>15}")
    for k in range(L):
        h = sp.to_gpu(A[jp], "h", k)
        line = f"  {k:5d} {h.norm(dim=-1).mean():10.2f}"
        if k:
            d, prev = sp.to_gpu(A[jp], "d", k), sp.to_gpu(A[jp], "h", k - 1)
            cos = torch.nn.functional.cosine_similarity(d, prev, dim=-1).mean()
            line += f" {d.norm(dim=-1).mean():10.2f} {(d.norm(dim=-1) / h.norm(dim=-1)).mean():8.3f} {cos:15.3f}"
        print(line)

    # ---------------------------------------------------------------------------------
    title("6  probe mechanics on these 5 rows (train = eval = all 5; accuracy must be 1.0)")
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    classes, y = np.unique([r["animal"] for r in rows], return_inverse=True)
    print(f"  classes {[str(c) for c in classes]} -> y {y}")
    all5 = np.arange(len(rows))
    parts = {"train": all5, "test": all5}
    feats = [("h", L - 1), ("d", L - 1), ("h", 0)]
    X = torch.stack([sp.to_gpu(A[jp], kind, k) for kind, k in feats])
    print(f"  X {tuple(X.shape)} = (probes fitted together, rows, hidden)")
    res = sp.fit(X, y, parts, len(classes))
    for (kind, k), r in zip(feats[:2], res):
        Xk = X[feats.index((kind, k))].cpu().numpy()
        skl = make_pipeline(StandardScaler(), PCA(min(sp.PCA_DIM, *Xk.shape)),
                            LogisticRegression(C=sp.C, max_iter=2000)).fit(Xk, y)
        print(f"  {kind}{k:02d}: ours train {r['train'][0]:.2f} | sklearn train {skl.score(Xk, y):.2f}")
        ok(r["train"][0] == 1.0, f"{kind}{k:02d} separates 5 rows")
    # h00 at assistant_tmp_end is the same vector in every row: only the prior is left
    null = np.bincount(y).max() / len(y)
    print(f"  h00: ours train {res[2]['train'][0]:.2f} (constant input, expected majority share {null:.2f})")
    ok(abs(res[2]["train"][0] - null) < 1e-9, "h00 null probe falls back to one class")

    # ---------------------------------------------------------------------------------
    title("7  splits on the full table (no model involved)")
    classes, y = np.unique([r["animal"] for r in all_rows], return_inverse=True)
    conds = [(n, idx) for n, _, idx in data.conditions(all_rows, seed=sp.SEED)]
    n_match = min(len(idx["train"]) for _, idx in conds)
    print(f"  {len(all_rows)} rows, {len(classes)} classes, n_match {n_match}")
    for name, parts in sp.restrict(conds, y, n_match, np.arange(len(all_rows))):
        print(f"  {name:10} " + "  ".join(f"{k} {len(v):5d}" for k, v in parts.items()))
    longest = Counter()
    for r in all_rows:
        for c in r["coords"]:
            seg, k = c.rsplit("_", 1)
            longest[seg] = max(longest[seg], int(k) + 1)
    print(f"  longest segments: {dict(longest)}")
    for coord in ("id_00", f"id_{longest['id'] // 2:02d}", f"id_{longest['id'] - 1:02d}"):
        avail = np.array([i for i, r in enumerate(all_rows) if coord in r["coords"]])
        cut = sp.restrict(conds, y, n_match, avail)
        print(f"\n  {coord}: coverage {len(avail) / len(all_rows):.3f} ({len(avail)} rows), "
              f"animals present {len(np.unique(y[avail]))}")
        if not cut:
            print(f"    every condition skipped (< {sp.MIN_TRAIN} train rows)")
        for name, parts in cut:
            print(f"    {name:10} " + "  ".join(f"{k} {len(v):5d}" for k, v in parts.items()))
    print("\ndone. Anything marked FAIL above needs a look before the full run.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="5 random rows instead of curated ones")

    original_stdout = sys.stdout

    with open("check_positions.md", "w", encoding="utf-8") as f:
        sys.stdout = Tee(original_stdout, f)
        try:
            main(ap.parse_args().seed)
        finally:
            sys.stdout = original_stdout