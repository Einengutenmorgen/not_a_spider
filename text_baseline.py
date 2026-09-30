"""Text-only baseline: the same conditions, but the feature is the prompt string.

If TF-IDF over the question text already separates the 24 animals under the
identifier split, then the identifiers of one animal share enough vocabulary
that the task is lexically solvable, and a high probe accuracy is not by itself
evidence about the model's internal representation. This is the number every
layer result has to be read against.

Not an entry in the classifier comparison and never a point on the layer curve:
it is a different feature space answering a different question, so it belongs as
one baseline row per condition. If it matches the probe on accuracy, compare the
error sets (predictions.csv, joinable on uid) -- disjoint errors at equal
accuracy mean the representation is doing something other than lexical matching.

    python text_baseline.py                  # word + char_wb
    python text_baseline.py --field full     # including the chat template
    LEAK_ACTION=skip python text_baseline.py # non-interactive leak handling
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

import data


SEED = 0
# word n-grams understate overlap between morphological variants across
# identifiers; char_wb catches those, so both are reported.
ANALYZERS = {"word": (1, 2), "char_wb": (3, 5)}


def build(analyzer, ngram, C=1.0):
    return make_pipeline(
        TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2,
                        sublinear_tf=True),
        LogisticRegression(C=C, max_iter=2000))


def find_leaks(classes, txt, tab):
    """Rows whose prompt contains their own animal's name, grouped by the wording
    they come from. A word-boundary hit names the animal outright; a substring-only
    hit is morphological ("camelid", "skunky"), which char_wb reads as the name
    just the same. Returns {(animal, ident, wording): [n_rows, exact, example]}."""
    bad = {}
    for c in classes:
        wb = re.compile(rf"\b{re.escape(c)}\b", re.I)
        for t, r in zip(txt, tab):
            if r["animal"] != c or c.lower() not in t.lower():
                continue
            key = (c, r["ident"], r["wording"])
            if key not in bad:
                bad[key] = [0, bool(wb.search(t)), t]
            bad[key][0] += 1
    return bad


def resolve_leaks(bad, tab):
    """Report the offending wordings and decide what to do. Returns the row indices
    to drop (empty when they are kept)."""
    print(f"\nanimal name appears in its own prompts: {len(bad)} wording(s), "
          f"{sum(v[0] for v in bad.values())} rows")
    for (c, ident, w), (n, exact, ex) in sorted(bad.items()):
        kind = "NAMES IT" if exact else "substring"
        print(f"  {c:<10} {ident:<9} wording {w:<2} {n:>6} rows  [{kind}]  {ex[:80]}")
    print("\nto drop these permanently, add to data.py:")
    print("  DROP_WORDINGS |= {" + ", ".join(
        f'("{c}", "{i}", {w})' for c, i, w in sorted(bad)) + "}")

    action = (os.environ.get("LEAK_ACTION") or "").lower()
    if not action:
        if not sys.stdin.isatty():
            raise SystemExit("\nnon-interactive: set LEAK_ACTION=skip|ignore|stop")
        while True:
            action = input("\n  s) skip these rows   i) ignore and use them   q) stop"
                           "\nSelect [s/i/q]: ").strip().lower()
            if action in ("s", "i", "q", "skip", "ignore", "stop"):
                break
            print("Invalid selection.")

    if action in ("q", "stop"):
        raise SystemExit("stopped on leaking prompts")
    if action in ("i", "ignore"):
        print("continuing WITH the leaking rows -- this inflates the baseline")
        return set()
    drop = {i for i, r in enumerate(tab)
            if (r["animal"], r["ident"], r["wording"]) in bad}
    print(f"skipping {len(drop)} rows")
    return drop


def main(field="question", base="results", analyzers=tuple(ANALYZERS), C=1.0):
    out_dir = data.run_dir(base)
    print(f"-> {out_dir}")

    rows, tab = data.analysis_rows(data.build_table())
    classes = np.unique([r["animal"] for r in tab])
    txt = np.array([r[field] for r in tab], dtype=object)

    # An animal name inside its own prompt would make this trivially circular.
    leak_action = "none"
    bad = find_leaks(classes, txt, tab)
    if bad:
        drop = resolve_leaks(bad, tab)
        leak_action = "skip" if drop else "ignore"
        if drop:
            keep = np.array([i for i in range(len(tab)) if i not in drop])
            rows, tab = rows[keep], [tab[i] for i in keep]
            txt = txt[keep]
            assert len(np.unique([r["animal"] for r in tab])) == len(classes), \
                "skipping the leaking rows removed a whole class"

    classes, y = np.unique([r["animal"] for r in tab], return_inverse=True)
    uid = np.array([r["uid"] for r in tab])

    (out_dir / "config.json").write_text(json.dumps(
        dict(script="text_baseline", field=field, analyzers=list(analyzers),
             C=C, run_dir=str(data.RUN_DIR), seed=SEED, n_rows=len(tab),
             leaks=[list(k) for k in sorted(bad)], leak_action=leak_action), indent=2))

    conds = data.conditions(tab, seed=SEED)
    n_match = min(len(idx["train"]) for _, _, idx in conds)
    print(f"{len(tab)} rows, {len(classes)} classes, n_train matched to {n_match}")

    with open(out_dir / "text_baseline.csv", "w", newline="") as fh, \
            open(out_dir / "predictions.csv", "w", newline="") as pf:
        w, pw = csv.writer(fh), csv.writer(pf)
        w.writerow(["feature", "condition", "split", "n_train", "n_eval", "acc", "f1"])
        pw.writerow(["feature", "condition", "uid", "y_true", "y_pred"])

        for an in analyzers:
            name_f = f"tfidf-{an}"
            for name, _, idx in conds:
                tr = data.match_size(idx["train"], y, n_match, seed=SEED)
                m = build(an, ANALYZERS[an], C)
                m.fit(txt[tr], y[tr])
                for part, ev in [("train", tr), ("dev", idx["dev"]),
                                 ("test", idx["test"])]:
                    p = m.predict(txt[ev])
                    w.writerow([name_f, name, part, len(tr), len(ev),
                                f"{accuracy_score(y[ev], p):.4f}",
                                f"{f1_score(y[ev], p, average='macro'):.4f}"])
                    if part == "test":
                        for u, t, q in zip(uid[ev], y[ev], p):
                            pw.writerow([name_f, name, u, classes[t], classes[q]])
                        print(f"{name_f:<14} {name:<10} "
                              f"test acc {accuracy_score(y[ev], p):.3f}", flush=True)
                fh.flush()
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="question", choices=["question", "full"])
    ap.add_argument("--analyzer", default="both",
                    choices=["both", "word", "char_wb"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--base", default="results")
    a = ap.parse_args()
    main(a.field, a.base,
         tuple(ANALYZERS) if a.analyzer == "both" else (a.analyzer,), a.C)