import glob
import numpy as np
import data

u = np.load("results_segments/uids.npy")
for d in sorted(glob.glob("acts/*/")):
    t = data.build_table(d)
    if not t:
        print(d, "no rows")
        continue
    try:
        _, tab = data.analysis_rows(t)
    except KeyError as e:
        print(d, "old schema, missing field", e)
        continue
    same = len(tab) == len(u) and all(r["uid"] == x for r, x in zip(tab, u))
    print(d, "rows:", len(t), "after drops:", len(tab), "same as probes:", same)