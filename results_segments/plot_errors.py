"""Error comparison of probe and text baseline over the token positions.

    python3 plot_errors.py [h|d] [matched|input_word|input_char_wb|full_word|full_char_wb]

Put it in results_segments/ with segments_common.py and run compare_errors.py --coords all first
(reads compare_errors.csv). 'matched' (default) compares positions that have read the answer with
full_word and all others with input_word. Writes
  figs/errors_delta_<family>_<baseline>.png    probe minus baseline accuracy on the same cases per position,
      with the 95% bootstrap interval over identifier clusters. green: p < 0.05 (probe better, best layer
      included in the test); blue: probe better but not significant; grey: probe not better; hollow
      marker: fewer than 30 clusters. Top panel counted from the segment start, bottom from the end.
  figs/errors_overlap_<family>_<baseline>.png  the same cases split by who is right, for key positions.
"""
import sys

import matplotlib
import numpy as np
import pandas as pd

import segments_common as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FAM = sys.argv[1] if len(sys.argv) > 1 else "h"
BASE = sys.argv[2] if len(sys.argv) > 2 else "matched"
KEY = ["end_of_ident", "after_ident", "end_of_question", "newline", "begin_assistant_formatting",
       "assistant_header_newline", "begin_of_answer", "end_of_answer"]
(sc.D / "figs").mkdir(exist_ok=True)

t = pd.read_csv(sc.D / "compare_errors.csv")
t = t[t["family"] == FAM]
if BASE == "matched":
    t = t[t["baseline"].to_numpy() == np.where([sc.sees_answer(c) for c in t["coord"]], "full_word", "input_word")]
else:
    t = t[t["baseline"] == BASE]
t = t.set_index("coord")

# ---- figure 1: difference per position ----
dense = t[t["n_clusters"] >= 30]
ymin, ymax = dense["ci_lo"].min() - 0.05, max(dense["ci_hi"].max(), 0) + 0.05
fig, axes = plt.subplots(2, 1, figsize=(28, 10), sharey=True)
for ax, sign in zip(axes, (1, -1)):
    cs = sc.order(set(t.index), sign)
    for i, c in enumerate(cs):
        if c not in t.index:
            continue
        r = t.loc[c]
        col = "tab:green" if r["p_maxT"] < 0.05 else ("tab:blue" if r["delta"] > 0 else "tab:gray")
        ax.errorbar(i, r["delta"], yerr=[[r["delta"] - r["ci_lo"]], [r["ci_hi"] - r["delta"]]], fmt="o", ms=4,
                    lw=1, color=col, mfc=col if r["n_clusters"] >= 30 else "none", clip_on=True)
    ax.axhline(0, color="k", lw=0.8)
    sc.segment_borders(ax, cs, ymax)
    ax.set_xlim(-1, len(cs))
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(range(len(cs)), cs, rotation=90, fontsize=5)
    ax.set_ylabel("probe - baseline accuracy")
    ax.set_title(f"{FAM} | baseline: {BASE} | {'counted from start' if sign > 0 else 'counted from end'}")
fig.legend([Line2D([], [], marker="o", color=c, ls="") for c in ("tab:green", "tab:blue", "tab:gray")] +
           [Line2D([], [], marker="o", color="k", mfc="none", ls="")],
           ["p < 0.05, probe better", "probe better, n.s.", "probe not better", "fewer than 30 clusters"],
           loc="upper center", ncol=4, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(sc.D / "figs" / f"errors_delta_{FAM}_{BASE}.png", dpi=120)
plt.close(fig)

# ---- figure 2: who is right, for key positions ----
d = t.loc[[c for c in KEY if c in t.index]]
share = d[["both", "probe_only", "base_only", "neither"]].div(d["n_test"], axis=0)
fig, ax = plt.subplots(figsize=(13, 0.7 * len(d) + 2))
left = np.zeros(len(d))
for col, color, lab in [("both", "#4c9f70", "both right"), ("probe_only", "#3b78c2", "only probe right"),
                        ("base_only", "#e08a2c", "only baseline right"), ("neither", "#c8c8c8", "both wrong")]:
    ax.barh(range(len(d)), share[col], left=left, color=color, label=lab)
    for i, (v, l) in enumerate(zip(share[col], left)):
        if v > 0.04:
            ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=9)
    left = left + share[col].to_numpy()
for i, (_, r) in enumerate(d.iterrows()):
    ax.text(1.01, i, f"probe {r['acc_probe']:.2f}  baseline {r['acc_base']:.2f}  "
                     f"{r['baseline']}", va="center", fontsize=9)
ax.set_yticks(range(len(d)), d.index)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("share of test cases")
ax.set_title(f"Who is right on the same test cases ({FAM}, best layer)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=4, fontsize=9)
fig.subplots_adjust(left=0.2, right=0.72, top=0.82)
fig.savefig(sc.D / "figs" / f"errors_overlap_{FAM}_{BASE}.png", dpi=120)
print(f"-> {sc.D / 'figs'}/errors_delta_{FAM}_{BASE}.png, errors_overlap_{FAM}_{BASE}.png")
