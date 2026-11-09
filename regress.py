"""Compare two runs of the indexing (e.g. the Olmo-7B reference against the template-agnostic scripts).

    python3 regress.py positions REF.jsonl NEW.jsonl    # every key of REF except shard/row, row by row, input_ids included
    python3 regress.py tokens REF.parquet NEW.parquet   # identical frames: values, columns, order
    python3 regress.py preds REF.npz NEW.npz            # one coordinate's probe predictions (results_segments/preds)
    python3 regress.py rebuild positions.jsonl DIR      # DIR/index_0.jsonl from a positions.jsonl (index fields only)
"""
import json
import sys
from itertools import zip_longest
from pathlib import Path


def positions(ref, new):
    n, bad, missing = 0, 0, 0
    with open(ref) as a, open(new) as b:
        for la, lb in zip_longest(a, b):
            if la is None or lb is None:
                missing += 1
                continue
            ra, rb = json.loads(la), json.loads(lb)
            n += 1
            diff = [k for k in ra if k not in ("shard", "row") and ra[k] != rb.get(k)]
            if diff and bad < 5:
                print(f"row {n} ({ra['uid']}): differs in {diff}")
            bad += bool(diff)
    print(f"{n} rows compared, {bad} differ, {missing} rows only in one of the files")
    return bad == 0 and missing == 0


def tokens(ref, new):
    import pandas as pd
    pd.testing.assert_frame_equal(pd.read_parquet(ref), pd.read_parquet(new))
    print("tokens.parquet identical")
    return True


def preds(ref, new):
    """Probe predictions of one coordinate: same test rows and token indices, agreement of the predicted classes."""
    import numpy as np
    a, b = np.load(ref, allow_pickle=True), np.load(new, allow_pickle=True)
    ok = list(a["features"]) == list(b["features"])
    for k in a.files:
        if k.endswith("_rows") or k.endswith("_t"):
            ok &= k in b.files and bool((a[k] == b[k]).all())
        elif k.endswith("_pred"):
            same = (a[k] == b[k]).mean(axis=1)                      # per feature
            print(f"{k[:-5]:10} predictions identical: mean {same.mean():.4%}, worst feature {same.min():.4%}")
    print("rows, t and features identical" if ok else "ROWS, T OR FEATURES DIFFER")
    return ok


def rebuild(src, out):
    from generate import FIELDS
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(src) as fh, open(Path(out) / "index_0.jsonl", "w") as o:
        for line in fh:
            r = json.loads(line)
            o.write(json.dumps({k: r[k] for k in FIELDS}) + "\n")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    ok = {"positions": positions, "tokens": tokens, "preds": preds, "rebuild": lambda *a: rebuild(*a) or True}[cmd](*args)
    sys.exit(0 if ok else 1)
