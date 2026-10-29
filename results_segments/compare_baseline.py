"""Probe vs text baseline on the strict condition, on pooled test cases.

    python3 compare_baseline.py

Put it in results_segments/ with segments_common.py (reads plan.csv, results/*.csv, preds/*.npz, uids.npy,
text_baseline.csv, text_baseline_predictions.csv there) and run it from anywhere; writes
  compare_strict.csv        one row per coordinate, the three strict folds pooled
  compare_strict_folds.csv  one row per coordinate and strict fold (accuracy of the layer chosen on the pooled cases)

Same definitions as plot_boxplots.py and compare_errors.py:
  - the probe layer of each family (d, h) is the one with the highest pooled accuracy over all test cases
  - the baselines are scored on exactly the cases of the coordinate, so they differ between coordinates
    that only a subset of the cases reaches (long identifiers or answers)
  - <name>_sd is the standard deviation of that accuracy over a bootstrap that resamples (animal,
    identifier) clusters, i.e. the standard error over identifiers; it is conditional on the chosen layer
Columns
  n_train, n_train_baseline   training cases of the probe / of the text baseline (the baseline trains on all)
  n_test, n_clusters          pooled test cases and (animal, identifier) clusters
  <baseline>                  TF-IDF on `input` (question up to the newline) or `full` (question + answer),
                              word or char_wb n-grams: input_word, input_char_wb, full_word, full_char_wb
  probe_d / probe_h, layer_d / layer_h
  gap_<d|h>_<baseline>        probe minus baseline (positive = probe better); the paired interval and the
                              p-value are in compare_errors.csv
Read `input` against identifier, question and template coordinates, `full` against answer coordinates.
"""
import numpy as np
import pandas as pd

import segments_common as sc

D = sc.D
BASE = sc.BASELINES
n_base = int(pd.read_csv(D / "text_baseline.csv")["n_train"].max())

rows, folds = [], []
for c in pd.read_csv(D / "plan.csv")["coord"]:
    cs = sc.load_cases(c)
    if cs is None:
        rows.append(dict(coord=c, note="skipped"))
        continue
    best = sc.best_layers(cs)
    V = np.vstack([cs["ok"][best["d"]], cs["ok"][best["h"]], sc.baseline_ok(cs)])    # [probe_d, probe_h, 4 baselines]
    names = ["probe_d", "probe_h", *BASE]
    acc, sd = V.mean(1), sc.cluster_boot(cs["cl"], V).std(1)
    r = pd.read_csv(D / "results" / f"{c}.csv")
    row = dict(coord=c, n_train=r[r["condition"].str.startswith("strict")]["n_train"].max(),
               n_train_baseline=n_base, n_test=V.shape[1], n_clusters=cs["cl"].max() + 1)
    for k, nm in enumerate(names[2:], 2):
        row[nm] = acc[k]
    for k, nm in enumerate(names[2:], 2):
        row[f"{nm}_sd"] = sd[k]
    for k, fam in enumerate("dh"):
        row[f"probe_{fam}"], row[f"probe_{fam}_sd"] = acc[k], sd[k]
        row[f"layer_{fam}"] = cs["feats"][best[fam]]
    for k, fam in enumerate("dh"):
        for i, b in enumerate(BASE):
            row[f"gap_{fam}_{b}"] = acc[k] - acc[2 + i]
    rows.append(row)
    for fold in sorted(set(cs["fold"])):
        m = cs["fold"] == fold
        folds.append(dict(coord=c, condition=fold, n_test=m.sum(), **{nm: V[k][m].mean() for k, nm in enumerate(names)}))

pd.DataFrame(rows).round(4).to_csv(D / "compare_strict.csv", index=False)
pd.DataFrame(folds).round(4).to_csv(D / "compare_strict_folds.csv", index=False)
print(pd.DataFrame(rows).round(3).head(8).to_string(index=False))
print(f"-> {D / 'compare_strict.csv'}, {D / 'compare_strict_folds.csv'}")