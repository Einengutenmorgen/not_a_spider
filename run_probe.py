"""Probe every hidden state and every residual update under the four conditions.

    python run_probe.py --features none --out controls.csv   # h[0] + length only
    python run_probe.py --only ident --out ident.csv
    python run_probe.py --stride 4                           # quick first pass

Every run writes into results/{counter}_{weekday}_{hour}_{minute}/ alongside a
config.json. n_train is matched across all four conditions regardless of --only,
so separate runs stay comparable.

L2 multinomial logistic regression is the instrument: linear, identical at every
layer, nothing selected per layer. --clf xgb is a side check for non-linear
structure on the dip and peak layers only, never the headline number.
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import data

PCA_DIM = 256
SEED = 0


def build(clf="logreg", C=1.0, pca=PCA_DIM, n_classes=24):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=pca, random_state=SEED))
    if clf == "logreg":
        steps.append(LogisticRegression(C=C, max_iter=2000))
    elif clf == "svc":
        steps.append(LinearSVC(C=C))
    elif clf == "xgb":
        from xgboost import XGBClassifier
        steps.append(XGBClassifier(objective="multi:softprob", num_class=n_classes,
                                   tree_method="hist", device="cuda",
                                   max_depth=6, learning_rate=0.1,
                                   n_estimators=400, random_state=SEED))
    return make_pipeline(*steps)


def score(model, X, y, tr, parts):
    """One fit, evaluated on train, dev and test. Returns metrics and predictions."""
    model.fit(X[tr], y[tr])
    out, pred = {}, {}
    for name, ev in parts.items():
        p = model.predict(X[ev])
        pred[name] = p
        out[name] = (accuracy_score(y[ev], p), f1_score(y[ev], p, average="macro"))
    return out, pred


def feature_names(features, stride, shard, world):
    names = []
    if features in ("all", "h"):
        names += [f"h{k:02d}" for k in data.STATES]
    if features in ("all", "d"):
        names += [f"d{k:02d}" for k in data.DELTAS]
    return names[::stride][shard::world]


def main(stride=1, clf="logreg", out="results.csv", only=None, features="all",
         shard=0, world=1, pca=PCA_DIM, C=1.0, predictions=False, base="results"):
    out_dir = data.run_dir(base)
    (out_dir / "config.json").write_text(json.dumps(
        dict(stride=stride, clf=clf, only=only, features=features, shard=shard,
             world=world, pca=pca, C=C, run_dir=str(data.RUN_DIR), seed=SEED),
        indent=2))
    print(f"-> {out_dir}")
    tab_all = data.build_table()
    rows, tab = data.analysis_rows(tab_all)
    classes, y = np.unique([r["animal"] for r in tab], return_inverse=True)
    conds = data.conditions(tab, seed=SEED)
    keep = [c for c in conds if only is None or c[0].startswith(only)]
    assert keep, f"no condition matches {only!r}"

    n_match = min(len(idx["train"]) for _, _, idx in conds)
    print(f"{len(tab)} rows, {len(classes)} classes, "
          f"n_train matched to {n_match} ({n_match / len(classes):.0f}/class)")
    for name, _, idx in keep:
        print(f"  {name:<10} train {len(idx['train']):>6}  dev {len(idx['dev']):>5}"
              f"  test {len(idx['test']):>6}")

    # h[0] is the embedding of the generation-prompt token: the same vector in
    # every row, so it cannot separate anything and a probe on it collapses to
    # the majority class. Fails loudly if rows are misaligned with the index.
    assert np.ptp(data.feature("h00", rows), axis=0).max() < 1e-6, "h[0] varies"

    # Prompt length as the only feature: a class-separable length distribution
    # would confound every result below.
    length = np.array([[r["n_tokens"]] for r in tab], float)

    feats = feature_names(features, stride, shard, world)
    jobs = ([("length", length)] if shard == 0 else []) + [(f, None) for f in feats]

    uid = np.array([r["uid"] for r in tab])

    # Resume: a feature is done when every kept condition has a test row for it.
    # Append whenever the file exists -- keying the mode on `done` would truncate a
    # CSV that exists but has no completed feature for the conditions asked for.
    csv_path = out_dir / out
    fresh = not csv_path.exists()
    done = set()
    if not fresh:
        with open(csv_path, newline="") as fh:
            seen = [(r["feature"], r["condition"]) for r in csv.DictReader(fh)
                    if r["split"] == "test"]
        want = {c[0] for c in keep}
        done = {f for f in {f for f, _ in seen}
                if want <= {c for g, c in seen if g == f}}
        jobs = [(f, X) for f, X in jobs if f not in done]
        print(f"resuming in {out_dir}: {len(done)} features done, {len(jobs)} to go")
        if not jobs:
            print("nothing left to do")
            return out_dir

    pf = open(out_dir / "predictions.csv", "w" if fresh else "a",
              newline="") if predictions else None
    if pf:
        pw = csv.writer(pf)
        if fresh:
            pw.writerow(["feature", "condition", "uid", "y_true", "y_pred"])

    with open(csv_path, "w" if fresh else "a", newline="") as fh:
        w = csv.writer(fh)
        if fresh:
            w.writerow(["feature", "condition", "split", "n_train", "n_eval", "acc", "f1"])
        for fname, X in jobs:
            if X is None:
                X = data.feature(fname, rows).astype(np.float32, copy=False)
                model_of = lambda: build(clf, C=C, pca=min(pca, X.shape[1]) if pca else 0,
                                         n_classes=len(classes))
            else:
                model_of = lambda: build("logreg", pca=0)
            for name, _, idx in keep:
                tr = data.match_size(idx["train"], y, n_match, seed=SEED)
                parts = {"train": tr, "dev": idx["dev"], "test": idx["test"]}
                got, pred = score(model_of(), X, y, tr, parts)
                if pf:   # test rows only: the full set would be ~2.5M lines
                    for u, t, p in zip(uid[idx["test"]], y[idx["test"]], pred["test"]):
                        pw.writerow([fname, name, u, classes[t], classes[p]])
                for part, (acc, f1) in got.items():
                    w.writerow([fname, name, part, len(tr), len(parts[part]),
                                f"{acc:.4f}", f"{f1:.4f}"])
                print(f"{fname} {name:<10} train {got['train'][0]:.3f}  "
                      f"test {got['test'][0]:.3f}", flush=True)
            fh.flush()
    if pf:
        pf.close()
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--clf", default="logreg", choices=["logreg", "svc", "xgb"])
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--only", default=None,
                    help="condition name prefix: ident, frame, slot, strict")
    ap.add_argument("--features", default="all", choices=["all", "h", "d", "none"],
                    help="hidden states, residual updates, both, or controls only")
    ap.add_argument("--shard", type=int, default=0, help="split features over runs")
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--pca", type=int, default=PCA_DIM, help="0 disables PCA")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--predictions", action="store_true",
                    help="write per-row test predictions for error analysis")
    ap.add_argument("--base", default="results")
    a = ap.parse_args()
    main(a.stride, a.clf, a.out, a.only, a.features, a.shard, a.world, a.pca, a.C,
         a.predictions, a.base)