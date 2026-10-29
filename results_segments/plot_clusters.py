"""Per-cluster view of one position: where does the probe beat the text baseline, and where not?

    python3 plot_clusters.py [coord] [h|d] [baseline] [subset]   # default: newline h input_word, all cases

Put it in results_segments/ with segments_common.py and run it from anywhere; writes
figs/clusters_<coord>_<family>_<baseline>[_<subset>].png. subset: correct, wrong or noncompliant
(case_subsets.csv from build_subsets.py); the layer is always the best one on all test cases. A cluster is one (animal, identifier) pair of the strict test
set (cases of all folds pooled); probe and baseline are scored on the same cases.
  left    heat map: rows = animals (sorted by mean difference), columns = the animal's held-out
          identifiers; colour = probe accuracy minus baseline accuracy on that cluster
  middle  one dot per cluster: baseline accuracy (x) against probe accuracy (y), size = number of cases;
          above the diagonal the probe is better
  right   all clusters sorted by the difference; the line and band are the pooled difference with its 95%
          bootstrap interval over clusters. If the bars are spread out and not carried by a few, the gain
          is not an artefact of a few identifiers.
"""
import sys

import matplotlib
import numpy as np
import pandas as pd

import segments_common as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COORD = sys.argv[1] if len(sys.argv) > 1 else "newline"
FAM = sys.argv[2] if len(sys.argv) > 2 else "h"
BASE = sys.argv[3] if len(sys.argv) > 3 else "input_word"
SUB = sys.argv[4] if len(sys.argv) > 4 else None
TAG = f"_{SUB}" if SUB else ""
(sc.D / "figs").mkdir(exist_ok=True)

full = sc.load_cases(COORD)
assert full is not None, f"{COORD}: no strict predictions"
j = sc.best_layers(full)[FAM]
cs = sc.load_cases(COORD, subset=SUB) if SUB else full
assert cs is not None, f"{COORD}: no cases in subset {SUB}"
cp, cb = cs["ok"][j], sc.baseline_ok(cs)[sc.BASELINES.index(BASE)]
rows = cs["rows"]
g = (pd.DataFrame(dict(animal=sc.animal[rows], ident=sc.ident[rows], probe=cp, base=cb))
     .groupby(["animal", "ident"]).agg(n=("probe", "size"), probe=("probe", "mean"), base=("base", "mean"))
     .reset_index())
g["delta"] = g["probe"] - g["base"]
ids = (pd.DataFrame(dict(animal=sc.animal[full["rows"]], ident=sc.ident[full["rows"]]))
       .drop_duplicates().sort_values(["animal", "ident"]))
ids["k"] = ids.groupby("animal").cumcount()                                 # 0..3: the animal's identifiers (all cases)
g = g.merge(ids, on=["animal", "ident"])
M = g.pivot(index="animal", columns="k", values="delta")
M = M.loc[M.mean(axis=1).sort_values(ascending=False).index]
delta = cp.mean() - cb.mean()
lo, hi = np.percentile(sc.cluster_boot(cs["cl"], (cp.astype(int) - cb)[None])[0], [2.5, 97.5])
lim = np.abs(g["delta"]).max()
cmap = plt.get_cmap("RdBu")                                                 # blue = probe better, red = baseline better
norm = plt.Normalize(-lim, lim)

fig, axes = plt.subplots(1, 3, figsize=(20, 7.5), gridspec_kw=dict(width_ratios=[1, 1, 1.4]))
ax = axes[0]
im = ax.imshow(M.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
for (r, k), v in np.ndenumerate(M.to_numpy()):
    if not np.isnan(v):
        ax.text(k, r, f"{v:+.2f}", ha="center", va="center", fontsize=8)
ax.set_yticks(range(len(M)), [f"{a} ({M.loc[a].mean():+.2f})" for a in M.index], fontsize=9)
ax.set_xticks(range(M.shape[1]), [f"identifier {k + 1}" for k in M.columns])
ax.set_title("probe - baseline accuracy per animal and identifier")
fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)

ax = axes[1]
ax.scatter(g["base"], g["probe"], s=20 + 160 * g["n"] / g["n"].max(), c=g["delta"], cmap=cmap, norm=norm,
           edgecolor="k", lw=0.4)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel(f"baseline accuracy ({BASE})")
ax.set_ylabel(f"probe accuracy ({FAM} {cs['feats'][j]})")
ax.set_title(f"clusters: probe better in {(g['delta'] > 0).sum()}, baseline better in {(g['delta'] < 0).sum()},"
             f" equal in {(g['delta'] == 0).sum()} of {len(g)}")

ax = axes[2]
s = g.sort_values("delta").reset_index(drop=True)
ax.bar(s.index, s["delta"], color=cmap(norm(s["delta"])), width=0.9)
ax.axhspan(lo, hi, color="k", alpha=0.12)
ax.axhline(delta, color="k", lw=1.2)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("clusters sorted by difference")
ax.set_ylabel("probe - baseline accuracy")
ax.set_title(f"pooled difference {delta:+.3f}, 95% CI {lo:+.3f} to {hi:+.3f}")
fig.suptitle(f"{COORD} | probe {FAM} {cs['feats'][j]} vs {BASE} | {SUB or 'all cases'}: {len(cp)} test cases, {len(g)} clusters")
fig.tight_layout()
fig.savefig(sc.D / "figs" / f"clusters_{COORD}_{FAM}_{BASE}{TAG}.png", dpi=120)
print(f"-> {sc.D / 'figs'}/clusters_{COORD}_{FAM}_{BASE}{TAG}.png")