"""Does the probe know anything TF-IDF doesn't? Item-level comparison on the test sets.

    python3 complement.py results_gpu/0928_143426/predictions.csv <text_run>/predictions.csv

Per condition (folds pooled; every test item sits in exactly one fold) x layer x text baseline:
    probe, text      test accuracy
    diff [lo, hi]    probe - text, 95% CI from resampling whole concepts (animal|ident)
    p_ok|t_wrong     share of the text baseline's errors the probe gets right
    t_ok|p_wrong     share of the probe's errors the text baseline gets right
    oracle           either one right: upper bound for any combination of the two

Reading: if errors were independent, p_ok|t_wrong would equal `probe`. Far below it means
the probe fails on the same items as text. diff CI above 0 means the probe beats text.
"""
import sys

import numpy as np
import pandas as pd

LAYERS = LAYERS = [
    "h00", "h01", "h02", "h03", "h04", "h05", "h06", "h07",
    "h08", "h09", "h10", "h11", "h12", "h13", "h14", "h15",
    "h16", "h17", "h18", "h19", "h20", "h21", "h22", "h23",
    "h24", "h25", "h26", "h27", "h28", "h29", "h30", "h31"
]
CONDITIONS = ["strict", "ident", "wording"]
N_BOOT = 2000
rng = np.random.default_rng(0)


def load(path, **filters):
    df = pd.read_csv(path)
    for col, val in filters.items():
        df = df[df[col] == val]
    df["cond"] = df.condition.str.split("/").str[0]
    df["ok"] = df.y_true == df.y_pred
    return df[df.cond.isin(CONDITIONS)]


def cluster_ci(values, clusters):
    """95% CI of mean(values), resampling whole clusters with replacement."""
    ids, idx = np.unique(clusters, return_inverse=True)
    sums, counts = np.bincount(idx, values), np.bincount(idx)
    draw = rng.integers(0, len(ids), (N_BOOT, len(ids)))
    return np.percentile(sums[draw].sum(1) / counts[draw].sum(1), [2.5, 97.5])


probe = load(sys.argv[1], pca=0)
text = load(sys.argv[2])

rows = []
for layer in LAYERS:
    for baseline in text.feature.unique():
        p_sub, t_sub = probe[probe.feature == layer], text[text.feature == baseline]
        both = p_sub.merge(t_sub, on=["cond", "condition", "uid"], suffixes=("_p", "_t"))
        assert len(both) == len(p_sub), f"uid mismatch for {layer} / {baseline}"

        for cond, g in both.groupby("cond"):
            p, t = g.ok_p.to_numpy(), g.ok_t.to_numpy()
            parts = g.uid.str.split("|")
            concept = parts.str[0] + "|" + parts.str[2]
            lo, hi = cluster_ci(p.astype(float) - t, concept)
            rows.append({"cond": cond, "text_baseline": baseline, "layer": layer, "n": len(g),
                         "concepts": concept.nunique(), "probe": p.mean(), "text": t.mean(),
                         "diff": p.mean() - t.mean(), "lo": lo, "hi": hi,
                         "p_ok|t_wrong": p[~t].mean(), "t_ok|p_wrong": t[~p].mean(),
                         "oracle": (p | t).mean()})

out = pd.DataFrame(rows).sort_values(["cond", "text_baseline", "layer"])
out.to_csv("complement.csv", index=False)
print(out.round(3).to_string(index=False))
