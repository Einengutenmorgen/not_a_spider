"""Share of the between-animal scatter that the top-k PCs of the training fold capture.

    ACTS=/media/data/hau/acts_local python3 scatter_share.py [h02 h21 ...]

Same preprocessing as the probe: standardise with train-fold stats, PCs from the train fold.
    between_train  share of the train-fold between-animal scatter inside the top-k PCs
    between_test   same for the held-out test rows, projected onto the train PCs
    total          share of total train variance inside the top-k PCs (PCA's explained variance)
between >> total means the animal signal sits in high-variance directions (PCA keeps it);
between_test well below 1 at k=256 means PCA drops animal signal the probe could have used.
"""
import csv
import sys

import numpy as np

import data

LAYERS = sys.argv[1:] or ["h00", "h02", "h11", "h15", "h17", "h19", "h21", "h24", "h26", "h30", "h31"]
KS = (64, 256, 1024)
SEED = 0


def between(Z, y):
    """Rows sqrt(n_c)(mu_c - mu): the squared Frobenius norm is the between-class scatter."""
    mu = Z.mean(0)
    return np.stack([np.sqrt((y == c).sum()) * (Z[y == c].mean(0) - mu) for c in np.unique(y)])


def share(B, P):
    total = np.linalg.norm(B) ** 2
    return np.linalg.norm(B @ P) ** 2 / total if total > 0 else float("nan")


tab_all = data.build_table()
rows, tab = data.analysis_rows(tab_all)
_, y = np.unique([r["animal"] for r in tab], return_inverse=True)
conds = data.conditions(tab, seed=SEED)
n_match = min(len(idx["train"]) for _, _, idx in conds)   # same train rows as run_probe

with open("scatter_share.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["feature", "condition", "k", "between_train", "between_test", "total"])
    for f in LAYERS:
        X = data.feature(f, rows)
        for name, _, idx in conds:
            tr, te = data.match_size(idx["train"], y, n_match, seed=SEED), idx["test"]
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1
            Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd

            ev, V = np.linalg.eigh((Ztr.T @ Ztr).astype(np.float64) / (len(tr) - 1))
            ev, V = ev[::-1], V[:, ::-1]                      # descending
            Btr, Bte = between(Ztr, y[tr]), between(Zte, y[te])

            for k in KS:
                P = V[:, :k]
                vals = (share(Btr, P), share(Bte, P), ev[:k].sum() / max(ev.sum(), 1e-12))
                w.writerow([f, name, k] + [f"{v:.4f}" for v in vals])
            print(f"{f} {name:<9} k=256  between_train {share(Btr, V[:, :256]):.3f}  "
                  f"between_test {share(Bte, V[:, :256]):.3f}  total {ev[:256].sum() / max(ev.sum(), 1e-12):.3f}",
                  flush=True)
        fh.flush()
