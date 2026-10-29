"""Per-case comparison of probe and text baseline on the strict condition, cluster-aware.

    python3 compare_errors.py [--coords newline begin_assistant_formatting | all] [--perm 10000]

Put it in results_segments/ with segments_common.py (reads uids.npy, plan.csv, preds/<coord>.npz and
text_baseline_predictions.csv there) and run it from anywhere; writes compare_errors.csv into the same folder.

Design: for every strict test case of a coordinate, the probe and the baseline either get the animal right or
wrong. Both are scored on exactly the same cases, pooled over the folds. Cases of one (animal, identifier) are
not independent, and the three strict folds share the same held-out identifiers, so the unit is the
(animal, identifier) cluster. The probe layer is the best one by pooled accuracy, as in plot_boxplots.py.
  delta      probe accuracy minus baseline accuracy on those cases
  ci         95% bootstrap interval of delta, resampling clusters (conditional on the chosen layer, so
             optimistic about the layer choice)
  p_maxT     one-sided p that the probe beats the baseline: the sign of each cluster's summed difference is
             flipped at random (--perm times); the statistic is the best delta over the 32 (h) or 31 (d)
             layers, so choosing the best layer is part of the test
  both / probe_only / base_only / neither   case counts; err_overlap = share of the probe's errors that
             the baseline also makes (high = same errors, low = a different error set)
"""
import argparse

import numpy as np
import pandas as pd

import segments_common as sc

ap = argparse.ArgumentParser()
ap.add_argument("--coords", nargs="+", default=["newline"], help="coordinate names, or 'all'")
ap.add_argument("--perm", type=int, default=10000)
args = ap.parse_args()
rng = np.random.default_rng(0)
coords = list(pd.read_csv(sc.D / "plan.csv")["coord"]) if args.coords == ["all"] else args.coords

out = []
for c in coords:
    cs = sc.load_cases(c)
    if cs is None:
        print(f"{c}: no strict predictions (skipped)")
        continue
    ok, cl, feats = cs["ok"], cs["cl"], cs["feats"]
    F, n = ok.shape
    nc = cl.max() + 1
    best = sc.best_layers(cs)
    for b, cb in zip(sc.BASELINES, sc.baseline_ok(cs)):
        diff = ok.astype(np.int8) - cb[None, :]                              # +1 probe only, -1 baseline only
        S = np.zeros((nc, F))
        np.add.at(S, cl, diff.T)                                             # summed difference per cluster
        delta = S.sum(0) / n
        flips = rng.choice([-1, 1], size=(args.perm, nc)) @ S / n            # sign-flip null, [perm, features]
        for fam in "hd":
            idx = [i for i, f in enumerate(feats) if f[0] == fam]
            f = best[fam]                                                    # = argmax delta over the family
            p = (1 + (flips[:, idx].max(1) >= delta[f]).sum()) / (args.perm + 1)
            lo, hi = np.percentile(sc.cluster_boot(cl, diff[f][None])[0], [2.5, 97.5])
            both, po, bo = (ok[f] & cb).sum(), (ok[f] & ~cb).sum(), (~ok[f] & cb).sum()
            neither = n - both - po - bo
            out.append(dict(coord=c, family=fam, baseline=b, layer=feats[f], n_test=n, n_clusters=nc,
                            acc_probe=ok[f].mean(), acc_base=cb.mean(), delta=delta[f], ci_lo=lo, ci_hi=hi,
                            p_maxT=p, both=both, probe_only=po, base_only=bo, neither=neither,
                            err_overlap=neither / max(neither + po, 1)))
    print(f"{c}: {n} cases, {nc} clusters", flush=True)

t = pd.DataFrame(out).round(4)
t.to_csv(sc.D / "compare_errors.csv", index=False)
print(t.drop(columns=["both", "probe_only", "base_only", "neither"]).to_string(index=False))
print(f"-> {sc.D / 'compare_errors.csv'}")