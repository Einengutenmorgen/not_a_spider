"""Merge results_segments/results/*.csv and draw position x layer heatmaps.

    python3 aggregate_segments.py [results_segments]

Writes into the results folder (D):
        all_results.csv   one row per (coord, family, layer, cond); slot/0..2 and strict/0..2 averaged
        best.csv          best layer per (coord, family, cond)
        figs/<cond>_<family>.png   top panel: coordinates counted from the segment start,
                                   bottom panel: counted from the segment end
family: h = hidden state, d = residual delta (h_k - h_k-1)
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(sys.argv[1] if len(sys.argv) > 1 else "results_segments")
CHANCE = 1 / 24
(D / "figs").mkdir(exist_ok=True)

df = pd.concat(pd.read_csv(f) for f in (D / "results").glob("*.csv"))
df["acc"] = pd.to_numeric(df["acc"], errors="coerce")          # "skipped" rows -> NaN
df["family"] = df["feature"].str[0]
df["layer"] = df["feature"].str[1:].astype(int)
df["cond"] = df["condition"].str.split("/").str[0]             # slot/0..2 -> slot, strict/0..2 -> strict
df = df.groupby(["coord", "family", "layer", "cond"], as_index=False)["acc"].mean()
df.to_csv(D / "all_results.csv", index=False)
ok = df.dropna()
ok.loc[ok.groupby(["coord", "family", "cond"])["acc"].idxmax()].to_csv(D / "best.csv", index=False)

coords = set(df["coord"])


def n_max(mid):
    return max(abs(int(c.rsplit("_", 1)[1])) for c in coords if c.startswith(mid + "_"))


def order(sign):
    """All coordinates in sequence order; sign=+1 names intermediates from the start, -1 from the end."""
    def seg(first, mid, last):
        n = n_max(mid)
        ks = range(1, n + 1) if sign > 0 else range(-n, 0)
        return [first, *[f"{mid}_{k}" for k in ks], last]
    return (seg("start_of_ident", "intermediate_ident", "end_of_ident")
            + seg("after_ident", "intermediate_question", "end_of_question")
            + ["end_user_formatting", "newline", "begin_assistant_formatting", "assistant_role",
               "assistant_header_newline"]
            + seg("begin_of_answer", "intermediate_answer", "end_of_answer")
            + ["end_assistant_formatting"])


for (cond, fam), g in df.groupby(["cond", "family"]):
    fig, axes = plt.subplots(2, 1, figsize=(28, 13))
    vmax = g["acc"].max()
    for ax, sign in zip(axes, (1, -1)):
        cs = order(sign)
        M = g.pivot(index="layer", columns="coord", values="acc").reindex(columns=cs)
        im = ax.imshow(M.values, origin="lower", aspect="auto", vmin=CHANCE, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(cs)), cs, rotation=90, fontsize=5)
        ax.set_yticks(range(len(M)), M.index, fontsize=6)
        ax.set_ylabel("layer")
        ax.set_title(f"{cond} | {'h (hidden)' if fam == 'h' else 'd (residual delta)'} | "
                     f"{'counted from start' if sign > 0 else 'counted from end'}")
        fig.colorbar(im, ax=ax, pad=0.01, label="test acc (chance 0.042)")
    fig.tight_layout()
    fig.savefig(D / "figs" / f"{cond}_{fam}.png", dpi=120)
    plt.close(fig)
print(f"{len(df)} rows -> {D}/all_results.csv, best.csv, figs/ ({df['cond'].nunique() * 2} figures)")