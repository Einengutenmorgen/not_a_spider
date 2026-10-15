"""Load the prompt_last activations, align them with the index, build splits.

Row order is fixed once by build_table(): rows.jsonl as written by fetch_layers.py,
which is the concatenation of index_0..index_N in filename order. Every array
downstream uses that order, and the h{kk}.npy files are already in it.

    ACTS=/media/data/hau/acts_local python run_probe.py --only strict
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

RUN_DIR = Path(os.environ.get("ACTS", "acts_local"))
LAYER_DIR = RUN_DIR               # fetch_layers.py writes h{kk}.npy at the top level

FINAL = 32                      # h[32] = norm(block 31): excluded from the sweep
STATES = list(range(32))        # h[0] .. h[31]
DELTAS = list(range(1, 32))     # d[k] = h[k] - h[k-1], one block's residual write

# Concepts whose wordings give the animal away as surface form. No word-boundary match
# names the animal, but a char n-gram model reads "camelid" as "camel", so the concept
# would be solvable from spelling alone rather than from what it denotes. Dropped at
# analysis time -- the rows stay in rows.jsonl and the activations are unaffected.
DROP_CONCEPTS = {("Camel", "ident_9")}   # "the largest living member of the camelid family"

# Single wordings that leak, where the rest of the concept is clean.
DROP_WORDINGS = {("Skunk", "ident_8", 2)}   # "...invoked for a skunky aroma in beer"
                                         # -- all ten wordings say "camelid"

# Single wordings that leak where the rest of the concept is clean. Cheaper than dropping
# the whole concept: the other nine wordings stay, which matters because the `wording`
# condition needs several per concept.
DROP_WORDINGS = {("Skunk", "ident_8", 2)}   # "...a skunky aroma in beer"


# --- reading ----------------------------------------------------------------

def build_table(run_dir=None):
    run_dir = Path(run_dir or RUN_DIR)
    src = run_dir / "rows.jsonl"
    if src.exists():
        return [json.loads(l) for l in src.read_text().splitlines()]
    rows = []                      # fall back to the raw index files
    for p in sorted(run_dir.glob("index_*.jsonl")):
        rows += [json.loads(l) for l in p.read_text().splitlines()]
    return rows


def feature(name, rows=None, layer_dir=None):
    """'h07' -> hidden state 7. 'd07' -> h7 - h6, the write of one block."""
    layer_dir = Path(layer_dir or LAYER_DIR)
    k = int(name[1:])

    def get(j):
        path = layer_dir / f"h{j:02d}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} -- refetch with: fetch_layers.py ... --layers {j}")
        a = np.load(path, mmap_mode="r")
        return np.asarray(a if rows is None else a[rows], dtype=np.float32)

    X = get(k)
    if name[0] == "d":
        X = X - get(k - 1)
    return X


def analysis_rows(tab, reliable_only=False):
    """Rows entering the conditions, as indices into the full table.

    v2 has no `kind` (jeopardy is gone) and no class-neutral identifier list, so
    taxonomic_class is an ordinary attribute: leakage already removed the
    (concept, attribute) pairs where the concept gives the answer away.
    DROP_CONCEPTS and DROP_WORDINGS remove text that leaks the animal as surface form.
    reliable_only is a no-op until the resolvability check has been rerun.
    """
    keep = [i for i, r in enumerate(tab)
            if (r["animal"], r["ident"]) not in DROP_CONCEPTS
            and (r["animal"], r["ident"], r["wording"]) not in DROP_WORDINGS
            and (r.get("reliable", True) or not reliable_only)]
    return np.array(keep), [tab[i] for i in keep]


# --- splits -----------------------------------------------------------------

# TODO(dev-split): dev is held out on the identifier axis only, so for the frame,
# slot and strict conditions dev is an easier distribution than test -- strict
# test came in ~0.16 below strict dev. Fine while the probe is a fixed
# instrument with nothing selected on dev; NOT fine once a configuration is
# chosen per condition. The v2 data has 10x the rows, so halving each
# condition's held-out values into dev and test is now affordable.

# TODO(clusters): errors cluster by concept, so the effective n for a CI is the
# number of held-out concepts (48 here), not the number of rows. Bootstrap over
# the `ident` key, and consider rotating every concept through test in folds.

AXES = {"ident": lambda r: f'{r["animal"]}|{r["ident"]}',
        "wording": lambda r: f'{r["animal"]}|{r["ident"]}|{r["wording"]}',
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


def holdout_wordings(tab, id_test, id_dev, n_test=2, seed=0):
    """Per concept, wordings to hold out -- the 'unseen wording of a SEEN concept'
    condition, so concepts already held out on the ident axis are skipped: their
    wordings are unseen anyway and would make this condition a copy of ident."""
    rng = np.random.default_rng(seed + 1)
    per = {}
    for r in tab:
        c = f'{r["animal"]}|{r["ident"]}'
        if c not in id_test and c not in id_dev:
            per.setdefault(c, set()).add(r["wording"])
    out = set()
    for c, ws in sorted(per.items()):
        ws = rng.permutation(sorted(ws))
        assert len(ws) > n_test, (c, len(ws))
        out |= {f"{c}|{w}" for w in ws[:n_test]}
    return out


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
    for part in ("train", "test"): #removed dev because it can be empty if n_dev=0
        assert len(idx[part]), f"{part} is empty"


def conditions(tab, n_frame_test=2, slot_groups=3, seed=0):
    """The conditions, as (name, constrained_axes, split). Slot and strict are
    folds: the slots are partitioned, each group is the test group once.

    wording holds out phrasings of concepts that ARE in train, so it isolates
    surface form from concept identity -- the axis the v2 rework was about.
    strict stays ident+frame+slot: a held-out concept already implies held-out
    wordings, so adding the axis there would change nothing."""
    rng = np.random.default_rng(seed)
    k = keyed = keys(tab)
    id_test, id_dev = holdout_identifiers(tab, n_test=4, n_dev=0, seed=seed)
    w_test = holdout_wordings(tab, id_test, id_dev, seed=seed)
    frames = rng.permutation(sorted(set(k["frame"])))
    f_test = set(frames[:n_frame_test])
    slots = rng.permutation(sorted(set(k["slot"])))
    groups = [set(g) for g in np.array_split(slots, slot_groups)]

    out = [("ident", ["ident"], split(tab, {"ident": id_test}, id_dev, keyed)),
           ("wording", ["wording"], split(tab, {"wording": w_test}, id_dev, keyed)),
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
    """Select an existing results directory or create a new one. RESULTS_DIR names
    one directly (that is how a non-interactive run resumes); RESULTS_NEW=1 or a
    non-interactive run without RESULTS_DIR always gets a fresh directory instead
    of blocking on input."""

    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)

    want = os.environ.get("RESULTS_DIR")
    if want:
        d = Path(want)
        if not d.is_absolute() and not d.exists():
            d = base / want
        d.mkdir(parents=True, exist_ok=True)
        return d

    dirs = sorted(
        [p for p in base.iterdir()
         if p.is_dir() and p.name.split("_")[0].isdigit()],
        key=lambda p: int(p.name.split("_")[0])
    )

    def fresh():
        n = max([int(p.name.split('_')[0]) for p in dirs], default=0) + 1
        d = base / f"{n:03d}_{datetime.now():%a_%H_%M}"
        d.mkdir()
        return d

    if os.environ.get("RESULTS_NEW") or not sys.stdin.isatty():
        return fresh()

    recent = dirs[-5:]
    print("\nExisting result directories:")
    for i, d in enumerate(recent, 1):
        print(f"  {i}) {d.name}")
    if not recent:
        print("  (none)")
    print("\n  n) Create new directory")

    while True:
        choice = input("\nSelect directory [1-5/n]: ").strip().lower()
        if choice == "n":
            return fresh()
        if choice.isdigit() and 1 <= int(choice) <= len(recent):
            return recent[int(choice) - 1]
        print("Invalid selection. Please enter a number or 'n'.")