"""Load the prompt_last activations, align them with the index, build splits.

Row order is fixed once by build_table(): the concatenation of index_0..index_N
in filename order. Every array downstream uses that order.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import ml_dtypes  # bfloat16 view; torch is not required anywhere in this file

RUN_DIR = Path("acts/20260914-151738-c26147")
LAYER_DIR = RUN_DIR / "layers"

FINAL = 32                      # h[32] = norm(block 31): excluded from the sweep
STATES = list(range(32))        # h[0] .. h[31]
DELTAS = list(range(1, 32))     # d[k] = h[k] - h[k-1], one block's residual write
DROP_SLOTS = {"taxonomic_class"}   # only slot built from class_neutral identifiers
DROP_KINDS = {"jeopardy"}


# --- reading ----------------------------------------------------------------

def st_open(path, key="prompt_last"):
    """Memory-map one tensor out of a safetensors file (header: u64 length, JSON,
    raw buffer). Avoids materialising a 540 MB shard and avoids a torch import."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(n))
    meta = head[key]
    dtype = {"BF16": ml_dtypes.bfloat16, "F16": np.float16,
             "F32": np.float32}[meta["dtype"]]
    start, end = meta["data_offsets"]
    buf = np.memmap(path, mode="r", dtype=np.uint8,
                    offset=8 + n + start, shape=(end - start,))
    return buf.view(dtype).reshape(meta["shape"])


def build_table(run_dir=None):
    run_dir = Path(run_dir or RUN_DIR)
    rows = []
    for p in sorted(run_dir.glob("index_*.jsonl")):
        rows += [json.loads(l) for l in p.read_text().splitlines()]
    return rows


def prepare(run_dir=None, chunk=500):
    """Shard-major -> layer-major: one .npy per hidden state, float16, in table
    order. Probing one layer would otherwise re-read all 5.9 GB of shards."""
    run_dir = Path(run_dir or RUN_DIR)
    tab = build_table(run_dir)
    out_dir = run_dir / "layers"
    out_dir.mkdir(exist_ok=True)

    by_shard = {}
    for i, r in enumerate(tab):
        by_shard.setdefault(r["shard"], []).append((r["row"], i))

    probe = st_open(run_dir / next(iter(by_shard)))
    n_hidden, d = probe.shape[1], probe.shape[2]
    out = [np.lib.format.open_memmap(out_dir / f"h{k:02d}.npy", mode="w+",
                                     dtype=np.float16, shape=(len(tab), d))
           for k in range(n_hidden)]

    for shard, pairs in by_shard.items():
        t = st_open(run_dir / shard)
        src = np.array([a for a, _ in pairs])
        dst = np.array([b for _, b in pairs])
        for i in range(0, len(src), chunk):
            block = t[src[i:i + chunk]].astype(np.float32)   # (m, n_hidden, d)
            for k in range(n_hidden):
                out[k][dst[i:i + chunk]] = block[:, k].astype(np.float16)
    for o in out:
        o.flush()
    (out_dir / "table.jsonl").write_text("\n".join(json.dumps(r) for r in tab))
    return tab


def feature(name, rows=None, layer_dir=None):
    """'h07' -> hidden state 7. 'd07' -> h7 - h6, the write of one block."""
    layer_dir = Path(layer_dir or LAYER_DIR)
    k = int(name[1:])

    def get(j):
        a = np.load(layer_dir / f"h{j:02d}.npy", mmap_mode="r")
        return np.asarray(a if rows is None else a[rows], dtype=np.float32)

    X = get(k)
    if name[0] == "d":
        X = X - get(k - 1)
    return X


def analysis_rows(tab):
    """Rows entering the four conditions, as indices into the full table."""
    keep = [i for i, r in enumerate(tab)
            if r["kind"] not in DROP_KINDS and r["slot"] not in DROP_SLOTS]
    return np.array(keep), [tab[i] for i in keep]


# --- splits -----------------------------------------------------------------

# TODO(dev-split): dev is held out on the identifier axis only, so for the frame,
# slot and strict conditions dev is an easier distribution than test -- strict
# test came in ~0.16 below strict dev. Fine while the probe is a fixed
# instrument with nothing selected on dev; NOT fine once a configuration is
# chosen per condition. The fix is to halve each condition's held-out values
# into dev and test (2 dev frames beside 2 test frames; 3 test / 2 dev / 4 train
# slots per fold), which drops strict training from ~5,950 to ~3,000 and, since
# everything is matched to the strict floor, shrinks all four conditions with it.
# That needs more usable data first: relax the identifier-reliability filter,
# extract more paraphrase frames, or re-admit taxonomic_class with a
# (animal, kind, ident) group key.

AXES = {"ident": lambda r: f'{r["animal"]}|{r["ident"]}',
        "frame": lambda r: str(r["frame"]),
        "slot": lambda r: r["slot"]}


def keys(tab):
    return {a: np.array([f(r) for r in tab]) for a, f in AXES.items()}


def holdout_identifiers(tab, n_test=2, n_dev=2, min_train=3, seed=0):
    """Per animal, since identifier indices are only unique within an animal."""
    rng = np.random.default_rng(seed)
    per = {}
    for r in tab:
        per.setdefault(r["animal"], set()).add(r["ident"])
    test, dev = set(), set()
    for animal, ids in sorted(per.items()):
        ids = rng.permutation(sorted(ids))
        assert len(ids) >= n_test + n_dev + min_train, (animal, len(ids))
        test |= {f"{animal}|{i}" for i in ids[:n_test]}
        dev |= {f"{animal}|{i}" for i in ids[n_test:n_test + n_dev]}
    return test, dev


def split(tab, test_vals, dev_idents, keyed=None):
    """A row is test only if it is held out on *every* constrained axis; one axis
    gives a single-axis condition, three give the strict condition. Train never
    contains a held-out value on any constrained axis, so a strict test row
    shares no identifier, no frame and no slot with anything trained on."""
    k = keyed or keys(tab)
    hit = {a: np.isin(k[a], list(v)) for a, v in test_vals.items()}
    is_test = np.logical_and.reduce(list(hit.values()))
    touched = np.logical_or.reduce(list(hit.values()))
    is_dev = np.isin(k["ident"], list(dev_idents))
    return {"train": np.where(~touched & ~is_dev)[0],
            "dev": np.where(~touched & is_dev)[0],
            "test": np.where(is_test & ~is_dev)[0]}


def check_split(tab, idx, axes, keyed=None):
    k = keyed or keys(tab)
    for a in axes:
        shared = set(k[a][idx["train"]]) & set(k[a][idx["test"]])
        assert not shared, f"{a}: {len(shared)} values shared train/test"
    assert not set(k["ident"][idx["train"]]) & set(k["ident"][idx["dev"]])


def conditions(tab, n_frame_test=2, slot_groups=3, seed=0):
    """The four conditions, as (name, constrained_axes, split). Slot and strict
    are folds: the slots are partitioned, each group is the test group once."""
    rng = np.random.default_rng(seed)
    k = keyed = keys(tab)
    id_test, id_dev = holdout_identifiers(tab, seed=seed)
    frames = rng.permutation(sorted(set(k["frame"])))
    f_test = set(frames[:n_frame_test])
    slots = rng.permutation(sorted(set(k["slot"])))
    groups = [set(g) for g in np.array_split(slots, slot_groups)]

    out = [("ident", ["ident"], split(tab, {"ident": id_test}, id_dev, keyed)),
           ("frame", ["frame"], split(tab, {"frame": f_test}, id_dev, keyed))]
    for i, g in enumerate(groups):
        out.append((f"slot/{i}", ["slot"], split(tab, {"slot": g}, id_dev, keyed)))
    for i, g in enumerate(groups):
        out.append((f"strict/{i}", ["ident", "frame", "slot"],
                    split(tab, {"ident": id_test, "frame": f_test, "slot": g},
                          id_dev, keyed)))
    for name, axes, idx in out:
        check_split(tab, idx, axes, keyed)
    return out


def match_size(idx_train, y, n, seed=0):
    """Equalise n_train across conditions. Structure (which identifiers, frames
    and slots are seen) is the variable under study, so rows are dropped at
    random within class rather than by group."""
    if n >= len(idx_train):
        return idx_train
    rng = np.random.default_rng(seed)
    per = max(1, n // len(np.unique(y[idx_train])))
    keep = []
    for c in np.unique(y[idx_train]):
        m = idx_train[y[idx_train] == c]
        keep.append(rng.choice(m, min(per, len(m)), replace=False))
    return np.sort(np.concatenate(keep))


def run_dir(base="results"):
    """Select an existing results directory or create a new one."""

    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)

    dirs = sorted(
        [p for p in base.iterdir()
         if p.is_dir() and p.name.split("_")[0].isdigit()],
        key=lambda p: int(p.name.split("_")[0])
    )

    # Show last 5 directories
    recent = dirs[-5:]

    print("\nExisting result directories:")
    if recent:
        for i, d in enumerate(recent, 1):
            print(f"  {i}) {d.name}")
    else:
        print("  (none)")

    print("\n  n) Create new directory")

    while True:
        choice = input("\nSelect directory [1-5/n]: ").strip().lower()

        if choice == "n":
            d = base / f"{max(
                [int(p.name.split('_')[0]) for p in dirs],
                default=0
            ) + 1:03d}_{datetime.now():%a_%H_%M}"

            d.mkdir()
            return d

        if choice.isdigit() and 1 <= int(choice) <= len(recent):
            return recent[int(choice) - 1]

        print("Invalid selection. Please enter a number or 'n'.")