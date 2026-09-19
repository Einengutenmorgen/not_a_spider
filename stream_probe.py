"""v2: probe every prompt position on the fly. No activations are written to disk.

    python stream_probe.py prepare                            # generate once -> runs/<RID>/table.jsonl
    RANK=r WORLD=4 python stream_probe.py probe --rid <RID>   # one process per GPU
    python stream_probe.py summary --rid <RID>                # h vs d, folds pooled

prepare   greedy answers are generated once and stored as token ids, together with the
          anchor positions of every row. Later passes are teacher-forced forwards over
          prompt + answer, so every position (answer tokens included) comes from one pass.
probe     positions are processed POS_PER_PASS at a time: one forward pass over all rows
          fills RAM with (positions x states x rows x d) float16, every probe for those
          positions is fitted on the GPU, one CSV per position is written, RAM is freed.
          Ranks split the positions, not the rows. A finished position is skipped on rerun.

Positions (docs/Probe_design_v2.md). Every token of every row is probed. Each token gets
a coordinate <segment>_<offset>, the offset counted from the start of its segment:
    sys   system turn + user header          identical in every row
    pre   question text before the identifier
    id    the identifier
    post  question text after the identifier
    tmpl  <|im_end|>\n<|im_start|>assistant\n  identical in every row
    ans   generated tokens, <|im_end|> included
A token belongs to the segment holding its last character. A coordinate is probed on the
rows that have it; positions.csv reports that share as coverage.

Markers name tokens for comparison:
    seq_begin            token 0
    ident_last           last identifier token
    end_user             the \n after the user's <|im_end|>
    assistant_tmp_begin  the <|im_start|> of the assistant header
    assistant_tmp_end    last prompt token; this is what v1 stored as prompt_last
                         (extract_acts_sharded.py reads h[0, -1] of the prefill pass)
    answer_first         first generated token
    answer_last          last generated token before <|im_end|>
    last                 final token of the sequence
A marker that sits on the same coordinate in every row (e.g. end_user = tmpl_01) is a label
on that coordinate. A marker that moves (ident_last, answer_last, last) is probed as its own
position.

Every position is probed on h00..h31 (hidden states) and d01..d31 (d[k] = h[k] - h[k-1],
the residual write of block k-1), with identical splits. h00 is the embedding of the input
token, i.e. the token-identity baseline for that position.

Instrument: the v1 pipeline on the GPU -- StandardScaler -> PCA(256) -> L2 multinomial
logistic regression (C=1), batched over LAYER_CHUNK features. Splits and matched n_train
come from data.py unchanged, so assistant_tmp_end (= v1 prompt_last)
should reproduce results/v.01.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from itertools import batched
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

import data  # conditions(), match_size(), DROP_SLOTS

MODEL = os.environ.get("MODEL", "allenai/Olmo-3-7B-Instruct")
DATA = Path("animals.json")
SELECTION = Path("identifier_selection.json")
OUT = Path("runs")
MAX_NEW_TOKENS = 48     # 8 cut most "number and unit" answers off while the model restated the subject
GEN_BATCH = 64
FWD_BATCH = 32
POS_PER_PASS = 4        # host RAM per pass: ~5 GB x this (32 states x 19.5k rows x 4096, fp16)
LAYER_CHUNK = 8         # probes fitted in parallel; GPU ~0.3 GB x this on top of the model
PCA_DIM = 256
C = 1.0
MAX_ITER = 1000
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD", 1))
MIN_TRAIN = 240         # a condition is skipped at a position with fewer training rows
SEGS = ["sys", "pre", "id", "post", "tmpl", "ans"]
MARKERS = ["seq_begin", "ident_last", "end_user", "assistant_tmp_begin",
           "assistant_tmp_end", "answer_first", "answer_last", "last"]
CAP_UNDEFINED = {"answer_last", "last"}   # in a capped row these sit on the cap, not the answer


def load(model=True):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if not model:
        return tok, None
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        attn_implementation="sdpa").to(DEVICE).eval()
    return tok, m


# --- prepare ----------------------------------------------------------------

def compose(frame, ident):
    q = frame.replace("{ID}", ident)
    return q[0].upper() + q[1:]


def coordinates(offsets, bounds):
    """Segment + offset for every token. bounds: [(segment, end_char), ...] in order;
    a token belongs to the segment holding its last character."""
    out, seen = [], Counter()
    for a, b in offsets:
        seg = next(name for name, end in bounds if max(b - 1, a) < end)
        out.append(f"{seg}_{seen[seg]:02d}")
        seen[seg] += 1
    return out


def build_rows(tok):
    """The v1 analysis rows (no jeopardy, no taxonomic_class, reliable identifiers
    only), with the question and identifier located in token space."""
    animals = json.loads(DATA.read_text())
    reliable = json.loads(SELECTION.read_text())["reliable"]
    rows = []
    for a in animals:
        keep = set(reliable[a["animal"]]["normal"])
        for slot, grid in a["attribute_questions_paraphrased"].items():
            if slot in data.DROP_SLOTS:
                continue
            for i, row in enumerate(grid):
                if row is None or i not in keep:
                    continue
                ident = a["identifier_variants"][i]
                for f, q in enumerate(row):
                    frame = a["question_frames"][slot][f]
                    assert q == compose(frame, ident), (a["animal"], slot, i, f)
                    msgs = [{"role": "user", "content": q}]
                    full = tok.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True)
                    head = len(tok.apply_chat_template(msgs, tokenize=False))
                    assert full.count(q) == 1 and head < len(full)
                    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
                    qs = full.index(q)
                    s = qs + frame.index("{ID}")
                    coords = coordinates(enc.offset_mapping, [
                        ("sys", qs), ("pre", s), ("id", s + len(ident)),
                        ("post", qs + len(q)), ("tmpl", len(full))])
                    # the generation prompt starts at `head`: the token before it closes the
                    # user turn, the token at it opens the assistant header
                    end_user = max(i for i, (_, b) in enumerate(enc.offset_mapping) if b <= head)
                    rows.append(dict(
                        uid=f"{a['animal']}|attribute|{slot}|{i}|{f}", animal=a["animal"],
                        kind="attribute", slot=slot, ident=i, frame=f, question=q,
                        n_tokens=len(enc.input_ids), coords=coords, seq_begin=0,
                        ident_last=max(k for k, c in enumerate(coords) if c.startswith("id_")),
                        end_user=end_user, assistant_tmp_begin=end_user + 1,
                        assistant_tmp_end=len(enc.input_ids) - 1,
                        input_ids=enc.input_ids))
    return rows


def normalize(text):
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def score(spec, answer):
    """Same rule as v1: word-boundary match for sets, first number in range (unit unchecked)."""
    if spec["type"] == "set":
        n = normalize(answer)
        return any(re.search(rf"\b{re.escape(normalize(c))}\b", n) for c in spec["accepted"])
    m = re.search(r"\d+(?:\.\d+)?", answer)
    return bool(m) and spec["min"] <= float(m.group()) <= spec["max"]


@torch.inference_mode()
def generate(model, tok, rows):
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    attrs = {a["animal"]: a["attributes"] for a in json.loads(DATA.read_text())}
    for b in range(0, len(rows), GEN_BATCH):
        chunk = rows[b:b + GEN_BATCH]
        T = max(r["n_tokens"] for r in chunk)
        ids = torch.full((len(chunk), T), pad)
        mask = torch.zeros_like(ids)
        for j, r in enumerate(chunk):                    # left padding for generate
            ids[j, T - r["n_tokens"]:] = torch.tensor(r["input_ids"])
            mask[j, T - r["n_tokens"]:] = 1
        out = model.generate(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE),
                             max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=pad)
        for r, g in zip(chunk, out[:, T:].tolist()):
            g = g[:next((k + 1 for k, t in enumerate(g) if t in eos), len(g))]
            content = g[:-1] if g[-1] in eos else g
            n = r["n_tokens"]
            r["coords"] = r["coords"][:n] + [f"ans_{j:02d}" for j in range(len(g))]
            r.update(gen_ids=g, answer=tok.decode(content, skip_special_tokens=True).strip(),
                     answer_first=n, answer_last=n + max(len(content), 1) - 1,
                     last=n + len(g) - 1)
            r["capped"] = g[-1] not in eos           # no end token: the answer may be cut off
            r["correct"] = score(attrs[r["animal"]][r["slot"]], r["answer"])
        print(f"generate {b + len(chunk)}/{len(rows)}", flush=True)


def prepare(rid=None):
    h = hashlib.sha1(f"{MODEL}|{MAX_NEW_TOKENS}".encode())   # a new cap is a new run
    for p in (DATA, SELECTION):
        h.update(p.read_bytes())
    rid = rid or f"v2-{h.hexdigest()[:8]}"
    run = OUT / rid
    if (run / "table.jsonl").exists():
        print(f"{run} exists, nothing to generate")
        return rid
    tok, model = load()
    rows = build_rows(tok)
    lens = [r["n_tokens"] for r in rows]
    print(f"{len(rows)} rows | prompt tokens {min(lens)}-{max(lens)} | "
          f"last prompt token {tok.decode(rows[0]['input_ids'][-1:])!r}")
    assert len({r["input_ids"][-1] for r in rows}) == 1, "prompt_last token differs"
    generate(model, tok, rows)
    ids = rows[0]["input_ids"] + rows[0]["gen_ids"]
    for m in MARKERS:                    # eyeball these on the real tokenizer
        i = rows[0][m]
        print(f"  {m:20s} {i:4d} {rows[0]['coords'][i]:8s} {tok.decode(ids[i:i + 1])!r}")
    run.mkdir(parents=True, exist_ok=True)
    (run / "table.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (run / "meta.json").write_text(json.dumps(dict(
        rid=rid, model=MODEL, max_new_tokens=MAX_NEW_TOKENS, n_rows=len(rows),
        accuracy=float(np.mean([r["correct"] for r in rows])), selection=str(SELECTION),
        capped=float(np.mean([r["capped"] for r in rows])),
        positions=positions(rows)[0], pca=PCA_DIM, C=C, seed=SEED), indent=2))
    capped = np.mean([r["capped"] for r in rows])
    print(f"accuracy {np.mean([r['correct'] for r in rows]):.3f} | "
          f"{capped:.1%} hit the {MAX_NEW_TOKENS}-token cap -> {run}")
    if capped > 0.01:
        print("  WARNING: answer_last / last / correct are unreliable for capped rows "
              "(flag 'capped' in table.jsonl); consider raising MAX_NEW_TOKENS")
    return rid


# --- probe ------------------------------------------------------------------

def positions(rows):
    """All coordinates in sequence order, then the markers that move between rows.
    Returns (names, aliases) with aliases[coordinate] = markers fixed on it."""
    for seg in ("sys", "tmpl"):
        texts = {tuple(t for t, c in zip(r["input_ids"], r["coords"]) if c.startswith(seg))
                 for r in rows}
        assert len(texts) == 1, f"{seg} tokens differ across rows"
    longest = Counter()
    for r in rows:
        for c in r["coords"]:
            seg, k = c.rsplit("_", 1)
            longest[seg] = max(longest[seg], int(k) + 1)
    names = [f"{seg}_{k:02d}" for seg in SEGS for k in range(longest[seg])]
    aliases = {}
    any_capped = any(r.get("capped") for r in rows)
    for m in MARKERS:
        at = {r["coords"][r[m]] for r in rows}
        # a marker undefined in capped rows is never a plain label: the coordinate would
        # include the capped rows, the marker does not
        if len(at) == 1 and not (any_capped and m in CAP_UNDEFINED):
            aliases.setdefault(at.pop(), []).append(m)
        else:
            names.append(m)
    return names, aliases


def locate(rows, names):
    """(rows, names) token index; -1 where the row has no such token."""
    out = np.full((len(rows), len(names)), -1)
    for j, r in enumerate(rows):
        at = {c: i for i, c in enumerate(r["coords"])}
        out[j] = [(-1 if r.get("capped") and p in CAP_UNDEFINED else r[p]) if p in MARKERS
                  else at.get(p, -1) for p in names]
    return out


def describe(rows, names, aliases, where, tok):
    """What each position is made of, for reading the results."""
    out = []
    for j, p in enumerate(names):
        toks = Counter((r["input_ids"] + r["gen_ids"])[i]
                       for r, i in zip(rows, where[:, j]) if i >= 0)
        top, n_top = toks.most_common(1)[0] if toks else (None, 0)
        n = sum(toks.values())
        out.append(dict(position=p, markers=" ".join(aliases.get(p, [p] if p in MARKERS else [])),
                        coverage=round(n / len(rows), 4), n_distinct_tokens=len(toks),
                        top_token="" if top is None else tok.decode([top]),
                        top_share=round(n_top / n, 3) if n else 0.0))
    return out


@torch.inference_mode()
def collect(model, rows, where, n_states, seed=SEED):
    """(positions, states, rows, d) float16 on the host. Every batch has the same shape and
    the order is shuffled: numerics must not depend on batch composition, or length (and
    so class) would leak into positions whose input is identical in every row."""
    seqs = [r["input_ids"] + r["gen_ids"] for r in rows]
    T = max(map(len, seqs))
    where = torch.as_tensor(where).clamp(min=0)      # missing rows are dropped before fitting
    out = np.empty((where.shape[1], n_states, len(rows), model.config.hidden_size), np.float16)
    order = np.random.default_rng(seed).permutation(len(rows))
    ar = torch.arange(FWD_BATCH, device=DEVICE)[:, None]
    for b in range(0, len(order), FWD_BATCH):
        idx = order[b:b + FWD_BATCH]
        full = np.concatenate([idx, np.repeat(idx[:1], FWD_BATCH - len(idx))])
        ids = torch.zeros(FWD_BATCH, T, dtype=torch.long)
        for j, i in enumerate(full):
            ids[j, :len(seqs[i])] = torch.tensor(seqs[i])
        # right padding under causal attention: pads never reach real tokens, no mask needed
        hs = model(ids.to(DEVICE), output_hidden_states=True, use_cache=False).hidden_states
        g = where[full].to(DEVICE)
        for s in range(n_states):        # hs[n_states] = norm(last block), excluded as in v1
            out[:, s, idx] = hs[s][ar, g][:len(idx)].transpose(0, 1).half().cpu().numpy()
        if b // FWD_BATCH % 100 == 0:
            print(f"rank {RANK}: forward {b + len(idx)}/{len(rows)}", flush=True)
    return out


def fit(X, y, parts, n_classes, pca=PCA_DIM):
    """Batched StandardScaler -> PCA -> L2 logistic regression. X: (c, n, d) on the GPU.
    The objective is separable across c, so one L-BFGS run solves all c probes.
    Returns, per probe, {part: (acc, macro_f1)}."""
    tr = parts["train"]
    Xt = X[:, tr]
    mu = Xt.mean(1, keepdim=True)
    sd = (Xt - mu).pow(2).mean(1, keepdim=True).sqrt()
    sd[sd < 10 * torch.finfo(sd.dtype).eps] = 1.0          # sklearn's zero-variance rule
    Z = (X - mu) / sd
    k = min(pca, *Xt.shape[1:])
    if pca and k:
        Z = Z @ torch.svd_lowrank(Z[:, tr], q=k, niter=4)[2]
    Zt = Z[:, tr]
    c, n, f = Zt.shape
    yt = torch.as_tensor(y[tr], device=X.device).repeat(c)
    W = torch.zeros(c, f, n_classes, device=X.device, requires_grad=True)
    b = torch.zeros(c, 1, n_classes, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=MAX_ITER, history_size=20,
                            tolerance_grad=1e-6, tolerance_change=1e-10,
                            line_search_fn="strong_wolfe")

    def closure():                       # = sklearn: sum(CE) + ||W||^2 / (2C), scaled by 1/n
        opt.zero_grad()
        loss = (F.cross_entropy((Zt @ W + b).reshape(-1, n_classes), yt, reduction="sum")
                + 0.5 / C * W.pow(2).sum()) / n
        loss.backward()
        return loss

    opt.step(closure)
    res = [{} for _ in range(c)]
    with torch.no_grad():
        for part, ev in parts.items():
            pred = (Z[:, ev] @ W + b).argmax(-1).cpu().numpy()
            for j in range(c):
                res[j][part] = ((pred[j] == y[ev]).mean(),
                                f1_score(y[ev], pred[j], average="macro"))
    return res


def to_gpu(A, kind, k):
    h = lambda j: torch.from_numpy(A[j]).to(DEVICE).float()
    return h(k) if kind == "h" else h(k) - h(k - 1)      # d[k]: the write of block k-1


def write_csv(path, header, lines):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(lines)
    tmp.rename(path)                     # a position counts as done only when complete


HEADER = ["position", "kind", "layer", "condition", "split", "n_train", "n_eval", "acc", "f1"]


def restrict(conds, y, n_match, avail):
    """The global splits cut down to the rows that have this position. Held-out
    identifiers, frames and slots stay the same; n_train is matched across the
    conditions that keep at least MIN_TRAIN rows. Full coverage gives exactly the
    global matched split."""
    cut = [(name, {k: np.intersect1d(v, avail) for k, v in idx.items()}) for name, idx in conds]
    cut = [(name, idx) for name, idx in cut
           if len(idx["train"]) >= MIN_TRAIN and len(idx["test"])]
    if not cut:
        return []
    n = min(n_match, *(len(idx["train"]) for _, idx in cut))
    return [(name, {"train": data.match_size(idx["train"], y, n, seed=SEED),
                    **{k: v for k, v in idx.items() if k != "train" and len(v)}})
            for name, idx in cut]


def probe_position(name, A, y, splits, n_classes, path):
    feats = [("h", k) for k in range(A.shape[0])] + [("d", k) for k in range(1, A.shape[0])]
    lines = []
    for chunk in batched(feats, LAYER_CHUNK):
        X = torch.stack([to_gpu(A, kind, k) for kind, k in chunk])
        for cname, parts in splits:
            for (kind, k), r in zip(chunk, fit(X, y, parts, n_classes)):
                lines += [[name, kind, k, cname, part, len(parts["train"]), len(parts[part]),
                           f"{acc:.4f}", f"{f1:.4f}"] for part, (acc, f1) in r.items()]
        del X
    write_csv(path, HEADER, lines)
    best = max(((float(l[7]), l[1], l[2]) for l in lines
                if l[3].startswith("strict") and l[4] == "test"), default=None)
    print(f"rank {RANK}: {name} done | " + (
        f"best strict fold test {best[0]:.3f} at {best[1]}{best[2]:02d}" if best
        else "no strict condition with enough rows"), flush=True)


def probe(rid):
    run = OUT / rid
    rows = [json.loads(l) for l in (run / "table.jsonl").read_text().splitlines()]
    classes, y = np.unique([r["animal"] for r in rows], return_inverse=True)
    conds = [(name, idx) for name, _, idx in data.conditions(rows, seed=SEED)]
    n_match = min(len(idx["train"]) for _, idx in conds)
    names, aliases = positions(rows)
    where = locate(rows, names)
    out = run / "probe"
    out.mkdir(exist_ok=True)
    tok, _ = load(model=False)
    info = {d["position"]: d for d in describe(rows, names, aliases, where, tok)}

    if RANK == 0:
        write_csv(run / "positions.csv", list(info[names[0]]),
                  [list(d.values()) for d in info.values()])
        full = restrict(conds, y, n_match, np.arange(len(rows)))
        length = torch.tensor([[[r["n_tokens"]]] for r in rows], dtype=torch.float32,
                              device=DEVICE).transpose(0, 1)
        write_csv(run / "controls.csv", HEADER, [
            ["-", "length", 0, cname, part, len(parts["train"]), len(parts[part]),
             f"{acc:.4f}", f"{f1:.4f}"]
            for cname, parts in full
            for part, (acc, f1) in fit(length, y, parts, len(classes), pca=0)[0].items()])

    mine = [j for j in range(len(names))[RANK::WORLD]
            if not (out / f"{names[j]}.csv").exists()]
    for j in [j for j in mine if (where[:, j] >= 0).sum() < MIN_TRAIN]:
        write_csv(out / f"{names[j]}.csv", HEADER, [])     # empty = too few rows, not failed
        print(f"rank {RANK}: {names[j]} skipped, {(where[:, j] >= 0).sum()} rows", flush=True)
    mine = [j for j in mine if (where[:, j] >= 0).sum() >= MIN_TRAIN]
    print(f"rank {RANK}/{WORLD}: {len(rows)} rows, n_train {n_match}, "
          f"{len(mine)}/{len(names)} positions to do", flush=True)
    if not mine:
        return
    _, model = load()
    n_states = model.config.num_hidden_layers            # h[0] .. h[L-1]
    for group in batched(mine, POS_PER_PASS):
        A = collect(model, rows, where[:, group], n_states)
        for a, j in enumerate(group):
            p, avail = names[j], np.where(where[:, j] >= 0)[0]
            # alignment: where every available row feeds the same token, h[0] is identical
            if info[p]["n_distinct_tokens"] == 1:
                assert np.ptp(A[a, 0, avail], axis=0).max() == 0, f"{p}: h[0] varies"
            if "seq_begin" in info[p]["markers"]:   # all zeros expected: fixed batch shape
                drift = np.ptp(A[a].astype(np.float32), axis=1).max(-1)
                print(f"rank {RANK}: {p} max spread per state "
                      f"{np.array2string(drift, precision=3, max_line_width=200)}")
            probe_position(p, A[a], y, restrict(conds, y, n_match, avail),
                           len(classes), out / f"{p}.csv")
        del A


# --- summary ----------------------------------------------------------------

def summary(rid):
    """Test accuracy per position x condition x layer, h and d side by side. Slot and
    strict folds are pooled weighted by n_eval (their test rows are disjoint)."""
    import pandas as pd
    run = OUT / rid
    df = pd.concat(pd.read_csv(p) for p in sorted((run / "probe").glob("*.csv")))
    df = df[df.split == "test"].assign(cond=lambda d: d.condition.str.split("/").str[0],
                                       hits=lambda d: d.acc * d.n_eval)
    g = df.groupby(["position", "cond", "layer", "kind"])[["hits", "n_eval"]].sum()
    acc = (g.hits / g.n_eval).unstack("kind")
    acc["d_minus_h"] = acc["d"] - acc["h"]
    pos = pd.read_csv(run / "positions.csv", keep_default_na=False).set_index("position")
    acc = acc.reset_index().join(pos[["markers", "coverage"]], on="position")
    acc.to_csv(run / "summary.csv", index=False)
    order = {p: i for i, p in enumerate(pos.index)}
    best = (acc[acc.markers != ""].groupby(["position", "markers", "cond"])[["h", "d"]]
            .max().unstack("cond"))
    print("best layer per marked position (all positions are in summary.csv)")
    print(best.sort_index(key=lambda s: s.map(order) if s.name == "position" else s)
          .round(3).to_string())
    print(f"-> {run / 'summary.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "probe", "summary"])
    ap.add_argument("--rid")
    a = ap.parse_args()
    if a.stage == "prepare":
        print(prepare(a.rid))            # last line of stdout is the RID
    else:
        assert a.rid, "--rid is required"
        {"probe": probe, "summary": summary}[a.stage](a.rid)