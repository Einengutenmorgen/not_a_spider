"""Probe vs text baseline on subsets of the strict test cases (correct / wrong / format not fulfilled).

    python3 compare_subsets.py [--coords newline ... | all] [--family h|d] [--baseline matched|input_word|..]
                               [--perm 10000]

Put it in results_segments/ with segments_common.py and case_subsets.csv (build_subsets.py). Writes
compare_subsets.csv and figs/subsets_<family>.png.

Only the evaluation is split: probes and baselines were trained on all training cases, nothing is retrained.
For each coordinate the probe layer is fixed to the best one on all test cases ("all"), so differences between
the subsets come from the cases, not from a different layer. Probe and baseline are scored on the same cases.
  majority    share of the most frequent animal in the subset (a subset can be unbalanced, so chance is not 1/24)
  delta, ci   probe minus baseline accuracy, 95% bootstrap interval over (animal, identifier) clusters
  p           one-sided sign-flip test over clusters, probe better (fixed layer, so no correction for the layer)
Before the answer starts (up to assistant_header_newline) the activations of a case do not depend on the answer
it later gets, so a difference between subsets there is a difference between the cases, not between model states.
Subsets with few clusters have wide intervals; hollow markers in the figure: fewer than 30 clusters.
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd

import segments_common as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--coords", nargs="+", default=["end_of_ident", "after_ident", "end_of_question", "newline",
                                                "begin_assistant_formatting", "assistant_header_newline",
                                                "begin_of_answer", "end_of_answer"])
ap.add_argument("--family", default="h", choices=["h", "d"])
ap.add_argument("--baseline", default="matched")
ap.add_argument("--perm", type=int, default=10000)
args = ap.parse_args()
rng = np.random.default_rng(0)
coords = list(pd.read_csv(sc.D / "plan.csv")["coord"]) if args.coords == ["all"] else args.coords
subsets = ["all", *sorted(set(pd.read_csv(sc.D / "case_subsets.csv")["subset"]))]

out = []
for c in coords:
    full = sc.load_cases(c)
    if full is None:
        continue
    j = sc.best_layers(full)[args.family]
    b = args.baseline if args.baseline != "matched" else ("full_word" if sc.sees_answer(c) else "input_word")
    for s in subsets:
        cs = full if s == "all" else sc.load_cases(c, subset=s)
        if cs is None:
            continue
        ok, cl = cs["ok"][j], cs["cl"]
        cb = sc.baseline_ok(cs)[sc.BASELINES.index(b)]
        n, nc = len(ok), cl.max() + 1
        diff = ok.astype(int) - cb
        S = np.bincount(cl, weights=diff)                                    # summed difference per cluster
        delta = S.sum() / n
        p = (1 + ((rng.choice([-1, 1], size=(args.perm, nc)) @ S / n) >= delta).sum()) / (args.perm + 1)
        lo, hi = np.percentile(sc.cluster_boot(cl, diff[None])[0], [2.5, 97.5])
        out.append(dict(coord=c, family=args.family, layer=cs["feats"][j], baseline=b, subset=s, n_test=n,
                        n_clusters=nc, majority=np.bincount(cs["y"]).max() / n, acc_probe=ok.mean(),
                        acc_base=cb.mean(), delta=delta, ci_lo=lo, ci_hi=hi, p=p,
                        probe_only=int((ok & ~cb).sum()), base_only=int((~ok & cb).sum())))
    print(f"{c}: done", flush=True)

t = pd.DataFrame(out).round(4)
t.to_csv(sc.D / "compare_subsets.csv", index=False)
print(t.drop(columns=["family", "probe_only", "base_only"]).to_string(index=False))

# ---- figure: difference per coordinate and subset ----
(sc.D / "figs").mkdir(exist_ok=True)
colors = dict(all="k", correct="tab:green", wrong="tab:red", noncompliant="tab:orange")
cp = [c for c in coords if c in set(t["coord"])]
fig, ax = plt.subplots(figsize=(11, 0.9 * len(cp) + 2))
for i, c in enumerate(cp):
    for k, s in enumerate(subsets):
        r = t[(t["coord"] == c) & (t["subset"] == s)]
        if r.empty:
            continue
        r = r.iloc[0]
        col = colors.get(s, "gray")
        ax.errorbar(r["delta"], i + (k - (len(subsets) - 1) / 2) * 0.16, fmt="o", ms=5, lw=1.2, color=col,
                    xerr=[[r["delta"] - r["ci_lo"]], [r["ci_hi"] - r["delta"]]],
                    mfc=col if r["n_clusters"] >= 30 else "none", label=s if i == 0 else None)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(range(len(cp)), cp)
ax.invert_yaxis()
ax.set_xlabel(f"probe - baseline accuracy on the same cases ({args.family}, layer fixed on all cases)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(subsets), fontsize=9)
fig.tight_layout()
fig.savefig(sc.D / "figs" / f"subsets_{args.family}.png", dpi=120)
print(f"-> {sc.D / 'compare_subsets.csv'}, {sc.D / 'figs'}/subsets_{args.family}.png")
