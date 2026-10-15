"""Significance table for the strict condition, one row per coordinate.

    python3 significance_strict.py [results_segments] [n_perm]

Needs results/<coord>.csv, preds/<coord>.npz, uids.npy, plan.csv (all written by run_segments.py).
Writes <results_segments>/strict_significance.csv.

Test: cases of one (animal, identifier) are not independent, so a plain binomial test on n_test would
be far too optimistic. Instead the animal labels are permuted across (fold, animal, identifier)
clusters while the probe's predictions stay fixed; the statistic is the best accuracy over the
layers of a family (h: 32 layers, d: 31), so the choice of the best layer is part of the test.
p = (1 + #{null max >= observed max}) / (n_perm + 1), one-sided, so p can not go below 1/(n_perm + 1);
z = (observed max - mean of null max) / sd of null max ranks the strong positions beyond that floor;
null95 = 95th percentile of the null max; n_idents = distinct (animal, identifier) pairs; p_bonf multiplies by the number of
tests (coordinates x 2 families). The three strict folds are pooled.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(sys.argv[1] if len(sys.argv) > 1 else "results_segments")
P = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
rng = np.random.default_rng(0)
uids = np.load(D / "uids.npy")
animal = np.array([u.split("|")[0] for u in uids])
ident = np.array([u.split("|")[2] for u in uids])


def test(z, k_names):
    """Pool the strict folds of one coordinate: labels, cluster ids, predictions [features, cases]."""
    classes = list(z["classes"])
    y, cl, pred, seen, ids = [], [], [], {}, set()
    for k in k_names:
        rows = z[f"{k}_rows"]
        y += [classes.index(a) for a in animal[rows]]
        cl += [seen.setdefault((k, a, i), len(seen)) for a, i in zip(animal[rows], ident[rows])]
        pred.append(z[f"{k}_pred"])
        ids |= set(zip(animal[rows], ident[rows]))
    return np.array(y), np.array(cl), np.concatenate(pred, axis=1), len(classes), len(ids)


def best_and_p(M, lab, idx, names, n):
    """M[cluster, predicted class, feature]; lab[cluster] = animal of the cluster."""
    ar = np.arange(len(lab))
    acc = M[ar, lab].sum(0) / n                                   # observed accuracy per feature
    null = np.array([M[ar, lab[rng.permutation(len(lab))]].sum(0) for _ in range(P)]) / n
    b = idx[acc[idx].argmax()]
    m = null[:, idx].max(1)
    return (names[b], acc[b], (1 + (m >= acc[b]).sum()) / (P + 1),
            (acc[b] - m.mean()) / m.std(), np.quantile(m, 0.95))


out = []
for c in pd.read_csv(D / "plan.csv")["coord"]:
    r = pd.read_csv(D / "results" / f"{c}.csv")
    r = r[r["condition"].str.startswith("strict")]
    row = {"coord": c, "n_train": r["n_train"].max()}
    z = np.load(D / "preds" / f"{c}.npz")
    k_names = sorted({f[:-5] for f in z.files if f.startswith("strict_") and f.endswith("_rows")})
    if not k_names:
        out.append({**row, "n_test": 0, "n_clusters": 0, "note": "skipped"})
        continue
    y, cl, pred, n_cls, n_ids = test(z, k_names)
    F, n = pred.shape
    nc = cl.max() + 1
    M = np.zeros((nc, n_cls, F), np.int32)                        # cluster x predicted class x feature
    np.add.at(M, (np.tile(cl, (F, 1)), pred.astype(int), np.arange(F)[:, None]), 1)
    lab = np.zeros(nc, int)
    lab[cl] = y
    names = [str(f) for f in z["features"]]
    row.update(n_test=n, n_clusters=nc, n_idents=n_ids, folds=len(k_names))
    for fam in "hd":
        idx = np.array([i for i, f in enumerate(names) if f[0] == fam])
        (row[f"{fam}_layer"], row[f"{fam}_acc"], row[f"{fam}_p"],
         row[f"{fam}_z"], row[f"{fam}_null95"]) = best_and_p(M, lab, idx, names, n)
    out.append(row)
    print(c, row["n_test"], f"h {row['h_acc']:.3f} z={row['h_z']:.1f}  d {row['d_acc']:.3f} z={row['d_z']:.1f}",
          flush=True)

t = pd.DataFrame(out)
n_tests = t["h_p"].notna().sum() + t["d_p"].notna().sum()
for fam in "hd":
    t[f"{fam}_p_bonf"] = (t[f"{fam}_p"] * n_tests).clip(upper=1)
t.to_csv(D / "strict_significance.csv", index=False)
print(f"-> {D / 'strict_significance.csv'}")