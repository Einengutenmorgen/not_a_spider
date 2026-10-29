"""Shared loading for the per-case analyses. Lives in results_segments/ next to the scripts that import it.

Everything is pooled over the folds of one condition (strict by default). A case is one test prompt; its
cluster is (animal, identifier), the unit that is resampled or permuted. The best layer of a family is the
one with the highest pooled accuracy; for a fixed baseline the same layer maximises the paired difference
probe - baseline, so every script picks the same layer.
"""
from functools import lru_cache
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent
uids = np.load(D / "uids.npy")
animal = np.array([u.split("|")[0] for u in uids])
ident = np.array([u.split("|")[2] for u in uids])
BASELINES = ["input_word", "input_char_wb", "full_word", "full_char_wb"]


@lru_cache(maxsize=None)
def _subsets():
    return pd.read_csv(D / "case_subsets.csv").set_index("uid")["subset"]


def load_cases(coord, cond="strict", subset=None):
    """Test cases of one coordinate pooled over the folds of `cond`, or None if it has none.
    subset: keep only the cases of this subset of case_subsets.csv (build_subsets.py).
    feats [F]; ok [F, n] probe correct; rows into uids.npy; fold name per case; cl cluster id per case."""
    z = np.load(D / "preds" / f"{coord}.npz")
    names = sorted(k[:-5] for k in z.files if k.endswith("_rows") and k.split("_")[0] == cond)   # strict_0 ..
    if not names:
        return None
    classes = list(z["classes"])
    rows = np.concatenate([z[f"{k}_rows"] for k in names])
    fold = np.concatenate([[k.replace("_", "/", 1)] * len(z[f"{k}_rows"]) for k in names])
    ok = np.concatenate([z[f"{k}_pred"] for k in names], axis=1) == \
        np.array([classes.index(a) for a in animal[rows]])[None, :]
    if subset is not None:
        keep = _subsets().reindex(uids[rows]).to_numpy() == subset
        if not keep.any():
            return None
        rows, fold, ok = rows[keep], fold[keep], ok[:, keep]
    return dict(feats=[str(f) for f in z["features"]], rows=rows, fold=fold, ok=ok,
                y=np.array([classes.index(a) for a in animal[rows]]),
                cl=pd.factorize(pd.Series(list(zip(animal[rows], ident[rows]))))[0])


def best_layers(c):
    """{"h": index, "d": index} of the layer with the highest pooled accuracy per family."""
    acc = c["ok"].mean(1)
    return {fam: max((i for i, f in enumerate(c["feats"]) if f[0] == fam), key=lambda i: acc[i]) for fam in "hd"}


@lru_cache(maxsize=None)
def _baseline_table(cond):
    bp = pd.read_csv(D / "text_baseline_predictions.csv")
    bp = bp[bp["condition"].str.split("/").str[0] == cond].copy()
    bp["ok"] = bp["y_true"] == bp["y_pred"]
    bp["base"] = bp["field"] + "_" + bp["feature"].str.replace("tfidf-", "", regex=False)
    return bp.pivot(index=["condition", "uid"], columns="base", values="ok")[BASELINES]


def baseline_ok(c, cond="strict"):
    """[4, n] bool: the text baselines on exactly the cases of c, rows in BASELINES order."""
    t = _baseline_table(cond).reindex(pd.MultiIndex.from_arrays([c["fold"], uids[c["rows"]]]))
    assert t.notna().all().all(), "test cases without a baseline prediction"
    return t.to_numpy(bool).T


def cluster_boot(cl, V, n_boot=2000, seed=0):
    """Bootstrap over clusters. V [k, n] per-case values; returns [k, n_boot] means of the resamples."""
    S = np.stack([np.bincount(cl, weights=v) for v in V])
    m = np.bincount(cl)
    b = np.random.default_rng(seed).integers(0, len(m), (n_boot, len(m)))
    return S[:, b].sum(2) / m[b].sum(1)


# ---- token positions in sequence order (for the figures) ----
def sees_answer(c):
    """The probe at this coordinate has read the generated answer."""
    return "answer" in c or c == "end_assistant_formatting"


def segment(c):
    if c in ("after_ident", "end_of_question") or "intermediate_question" in c:
        return "question"
    if "ident" in c:
        return "identifier"
    return "answer" if "answer" in c else "template"


def order(coords, sign):
    """All coordinates in sequence order; sign=+1 names intermediates from the start, -1 from the end."""
    def n_max(mid):
        return max(abs(int(c.rsplit("_", 1)[1])) for c in coords if c.startswith(mid + "_"))

    def seg(first, mid, last):
        n = n_max(mid)
        ks = range(1, n + 1) if sign > 0 else range(-n, 0)
        return [first, *[f"{mid}_{k}" for k in ks], last]
    return (seg("start_of_ident", "intermediate_ident", "end_of_ident")
            + seg("after_ident", "intermediate_question", "end_of_question")
            + ["end_user_formatting", "newline", "begin_assistant_formatting", "assistant_role",
               "assistant_header_newline"]
            + seg("begin_of_answer", "intermediate_answer", "end_of_answer")
            + ["end_assistant_formatting"])


def segment_borders(ax, cs, ytext):
    """Vertical lines between the segments of the ordered coordinates cs, and their names at height ytext."""
    i = 0
    for name, run in groupby(segment(c) for c in cs):
        n = len(list(run))
        if i:
            ax.axvline(i - 0.5, color="k", lw=0.6, alpha=0.4)
        ax.text(i + n / 2 - 0.5, ytext, name, ha="center", va="top", fontsize=8)
        i += n