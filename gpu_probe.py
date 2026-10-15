"""GPU version of run_probe.py for the PCA check: both arms with the same solver.

    CUDA_VISIBLE_DEVICES=3 ACTS=/media/data/hau/acts_local python3 -u gpu_probe.py
    ... gpu_probe.py --layers h21 h26 --pca 256 0 --only strict
    ... gpu_probe.py --coords prompt_last end_of_ident newline --pca 0

--coords: prompt_last reads the Path 1 activations (data.feature); any other position
reads $POS_DIR/<position>/hNN.npy written by extract_positions.py. dNN = hNN - h(N-1).

Same rows, splits, match_size, standardisation and objective as run_probe.py:
sklearn's LogisticRegression minimises C * sum(CE) + ||W||^2 / 2, which is
mean(CE) + ||W||^2 / (2 C n) below (bias unpenalised), solved with full-batch L-BFGS.
PCA is exact (eigh of the train covariance) instead of sklearn's randomised solver.

Sanity check: the pca=256 rows should reproduce results.csv to within ~0.01.
Writes results.csv (+ pca, n_iter, grad columns) and test predictions per fit.
"""
import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data

SEED = 0
POS_DIR = Path(os.environ.get("POS_DIR", "/media/data/hau/acts_pos"))
#LAYERS = ["h00", "h02", "h11", "h15", "h17", "h19", "h21", "h24", "h26", "h30", "h31"]
LAYERS = [
    "h00", "h01", "h02", "h03", "h04", "h05", "h06", "h07",
    "h08", "h09", "h10", "h11", "h12", "h13", "h14", "h15",
    "h16", "h17", "h18", "h19", "h20", "h21", "h22", "h23",
    "h24", "h25", "h26", "h27", "h28", "h29", "h30", "h31"
]

def load(coord, f, rows):
    """Activations of feature f at position coord, one row per analysis row."""
    if coord == "prompt_last":
        return data.feature(f, rows)
    h = lambda k: np.load(POS_DIR / coord / f"h{k:02d}.npy", mmap_mode="r").astype(np.float32)
    k = int(f[1:])
    return h(k) - h(k - 1) if f[0] == "d" else h(k)     # extract_positions: already in row order


def fit_logreg(Z, y, n_classes, C=1.0, max_iter=2000, tol=1e-5):
    W = torch.zeros(Z.shape[1], n_classes, device=Z.device, requires_grad=True)
    b = torch.zeros(n_classes, device=Z.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, history_size=20, tolerance_grad=tol,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")
    lam = 1.0 / (2 * C * len(y))

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Z @ W + b, y) + lam * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    grad = max(W.grad.abs().max().item(), b.grad.abs().max().item())
    return W.detach(), b.detach(), opt.state[W]["n_iter"], grad


def main(coords, layers, pcas, only, C, device):
    out_dir = Path("results_gpu") / datetime.now().strftime("%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    print(f"-> {out_dir}  on {torch.cuda.get_device_name(device) if 'cuda' in device else device}")

    rows, tab = data.analysis_rows(data.build_table())
    classes, y_np = np.unique([r["animal"] for r in tab], return_inverse=True)
    uid = np.array([r["uid"] for r in tab])
    conds = data.conditions(tab, seed=SEED)
    n_match = min(len(idx["train"]) for _, _, idx in conds)       # as in run_probe
    conds = [c for c in conds if only is None or c[0].startswith(only)]
    y = torch.as_tensor(y_np, device=device)

    res = open(out_dir / "results.csv", "w", newline="")
    prd = open(out_dir / "predictions.csv", "w", newline="")
    wr, wp = csv.writer(res), csv.writer(prd)
    wr.writerow(["coord", "feature", "condition", "pca", "split", "n_train", "n_eval", "acc", "f1",
                 "n_iter", "grad"])
    wp.writerow(["coord", "feature", "condition", "pca", "uid", "y_true", "y_pred"])

    for coord, f in [(c, f) for c in coords for f in layers]:
        X = torch.as_tensor(load(coord, f, rows), device=device)
        for name, _, idx in conds:
            tr = data.match_size(idx["train"], y_np, n_match, seed=SEED)
            parts = {"train": tr, "dev": idx["dev"], "test": idx["test"]}
            mu, sd = X[tr].mean(0), X[tr].std(0, unbiased=False)
            sd[sd == 0] = 1
            Z = (X - mu) / sd                                         # all rows, train stats
            ev, V = torch.linalg.eigh(Z[tr].T.double() @ Z[tr].double() / (len(tr) - 1))
            for k in pcas:
                P = V[:, -k:].float() if k else None                 # eigh is ascending
                Zk = Z @ P if k else Z
                W, b, n_iter, grad = fit_logreg(Zk[tr], y[tr], len(classes), C=C)
                for part, ev_idx in parts.items():
                    p = (Zk[ev_idx] @ W + b).argmax(1).cpu().numpy()
                    t = y_np[ev_idx]
                    acc = (p == t).mean()
                    f1 = np.mean([2 * ((p == c) & (t == c)).sum() / max((p == c).sum() + (t == c).sum(), 1)
                                  for c in range(len(classes))])     # macro-F1 over all classes
                    wr.writerow([coord, f, name, k, part, len(tr), len(ev_idx), f"{acc:.4f}", f"{f1:.4f}",
                                 n_iter, f"{grad:.1e}"])
                    if part == "test":
                        wp.writerows([coord, f, name, k, u, classes[a], classes[b_]]
                                     for u, a, b_ in zip(uid[ev_idx], t, p))
                        print(f"{coord} {f} {name:<9} pca={k:<4} test {acc:.3f}  iters {n_iter}  grad {grad:.1e}",
                              flush=True)
                res.flush(); prd.flush()
            del Z
        del X
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", nargs="+", default=["prompt_last"])
    ap.add_argument("--layers", nargs="+", default=LAYERS)
    ap.add_argument("--pca", nargs="+", type=int, default=[256, 0], help="0 = no PCA")
    ap.add_argument("--only", default=None, help="condition prefix, e.g. strict")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    main(a.coords, a.layers, a.pca, a.only, a.C, a.device)