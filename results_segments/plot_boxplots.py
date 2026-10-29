"""Box plots per token position, with the text baselines as horizontal lines.

    python3 plot_boxplots.py [cond] [cases|layers]     # cond: strict (default), slot, ident, wording, frame

Put it in results_segments/ with segments_common.py (reads preds/*.npz, uids.npy, all_results.csv and
text_baseline.csv there) and run it from anywhere; writes figs/box_<cond>_<mode>_<h|d>.png.

Per position, one box per family (h: hidden states, d: residual deltas):
  cases   (default) per-case predictions pooled over the folds; the best layer is the one with the
          highest pooled accuracy; line = that accuracy, box = 25-75% and whiskers = 2.5-97.5% of a
          bootstrap that resamples (animal, identifier) clusters (2000 draws). Conditional on the
          chosen layer, so slightly optimistic.
  layers  the spread over all layers of the family (folds averaged): box = quartiles, whiskers
          1.5 x IQR, black diamond = the best layer
Horizontal lines: the four text baselines (mean over all test cases of the condition, so at rare positions
they are not the baseline on the same cases, see compare_errors.csv) and chance (1/24).
Top panel: intermediate tokens counted from the segment start, bottom panel: from the end.
"""
import sys

import matplotlib
import numpy as np
import pandas as pd

import segments_common as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sc.D
COND = sys.argv[1] if len(sys.argv) > 1 else "strict"
MODE = sys.argv[2] if len(sys.argv) > 2 else "cases"
STYLE = {("input", "tfidf-word"): ("tab:red", "-"), ("input", "tfidf-char_wb"): ("tab:orange", "-"),
         ("full", "tfidf-word"): ("tab:blue", "--"), ("full", "tfidf-char_wb"): ("tab:cyan", "--")}
(D / "figs").mkdir(exist_ok=True)

if MODE == "layers":
    a = pd.read_csv(D / "all_results.csv")
    a = a[a["cond"] == COND].dropna()
    spread = {fam: {c: g["acc"].to_numpy() for c, g in a[a["family"] == fam].groupby("coord")} for fam in "hd"}
else:
    spread = {"h": {}, "d": {}}
    for f in (D / "preds").glob("*.npz"):
        c = sc.load_cases(f.stem, COND)
        if c is None:
            continue
        for fam, j in sc.best_layers(c).items():
            lo, q1, q3, hi = np.percentile(sc.cluster_boot(c["cl"], [c["ok"][j]])[0], [2.5, 25, 75, 97.5])
            spread[fam][f.stem] = dict(med=c["ok"][j].mean(), q1=q1, q3=q3, whislo=lo, whishi=hi, fliers=[])
tb = pd.read_csv(D / "text_baseline.csv")
tb = tb[(tb["split"] == "test") & (tb["condition"].str.split("/").str[0] == COND)]
base = tb.groupby(["field", "feature"])["acc"].mean()
coords = set(spread["h"]) | set(spread["d"])

top = max(max(v["whishi"] if isinstance(v, dict) else v.max() for g in spread.values() for v in g.values()),
          base.max()) * 1.1
for fam in "hd":
    groups = spread[fam]
    fig, axes = plt.subplots(2, 1, figsize=(28, 12), sharey=True)
    for ax, sign in zip(axes, (1, -1)):
        cs = sc.order(coords, sign)
        pos = [i for i, c in enumerate(cs) if c in groups]
        style = dict(widths=0.7, showfliers=False, patch_artist=True, boxprops=dict(facecolor="lightgray"),
                     medianprops=dict(color="k"))
        if MODE == "layers":
            ax.boxplot([groups[cs[i]] for i in pos], positions=pos, whis=1.5, **style)
            ax.scatter(pos, [groups[cs[i]].max() for i in pos], marker="D", s=12, color="k", zorder=3,
                       label="best layer")
        else:
            ax.bxp([groups[cs[i]] for i in pos], positions=pos, **style)
            ax.plot([], [], "k-", label="best-layer accuracy (box 25-75%, whiskers 2.5-97.5% bootstrap)")
        for (field, an), (col, ls) in STYLE.items():
            ax.axhline(base[(field, an)], color=col, ls=ls, lw=1.3, label=f"baseline {field} {an}")
        ax.axhline(1 / 24, color="gray", ls=":", lw=1, label="chance")
        sc.segment_borders(ax, cs, top)
        ax.set_xlim(-1, len(cs))
        ax.set_ylim(0, top)
        ax.set_xticks(range(len(cs)), cs, rotation=90, fontsize=5)
        ax.set_ylabel("test accuracy")
        ax.set_title(f"{COND} | {'h (hidden)' if fam == 'h' else 'd (residual delta)'} | {MODE} | "
                     f"{'counted from start' if sign > 0 else 'counted from end'}")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=7, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(D / "figs" / f"box_{COND}_{MODE}_{fam}.png", dpi=120)
    plt.close(fig)
print(f"-> {D / 'figs'}/box_{COND}_{MODE}_h.png, box_{COND}_{MODE}_d.png")