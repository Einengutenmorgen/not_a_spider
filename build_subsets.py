"""Splits the strict test cases by what the model answered: correct, wrong, or format not fulfilled.

    python3 build_subsets.py                          # lists the keys of positions.jsonl; all cases count as format-compliant
    python3 build_subsets.py --format-key NAME        # NAME: key of positions.jsonl, truthy = format fulfilled
    python3 build_subsets.py --format-key NAME --invert    # truthy = format violated

Run it where positions.jsonl lives (~/not_a_spider). Writes results_segments/case_subsets.csv with
uid, correct, format_ok, subset; the three subsets partition all cases:
  noncompliant   the format is not fulfilled (whatever the content)
  correct        format fulfilled and the answer is correct
  wrong          format fulfilled and the answer is wrong
`correct` is the key `correct` of positions.jsonl. Probes and baselines are not retrained: the subsets are only
used to evaluate the saved predictions (compare_subsets.py, plot_clusters.py ... --subset).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--format-key", default=None)
ap.add_argument("--invert", action="store_true")
ap.add_argument("--out", default="results_segments/case_subsets.csv")
args = ap.parse_args()

rows = []
with open("positions.jsonl") as fh:
    for i, line in enumerate(fh):
        r = json.loads(line)
        if i == 0:
            print("keys of positions.jsonl:", sorted(k for k in r if k != "input_ids"))
            if "correct" not in r or (args.format_key and args.format_key not in r):
                raise SystemExit("`correct` or the format key is missing in positions.jsonl, see the keys above")
        fmt = (bool(r[args.format_key]) != args.invert) if args.format_key else True
        rows.append((r["uid"], bool(r["correct"]), fmt,
                     "noncompliant" if not fmt else ("correct" if r["correct"] else "wrong")))
t = pd.DataFrame(rows, columns=["uid", "correct", "format_ok", "subset"])
Path(args.out).parent.mkdir(exist_ok=True)
t.to_csv(args.out, index=False)
print(t["subset"].value_counts().to_string())
print(f"-> {args.out}")
