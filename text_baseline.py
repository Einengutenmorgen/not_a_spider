"""Text-only baseline: the same conditions, but the feature is the text the probe could have seen.

    input   the question (identifier + attribute frame), i.e. everything up to the newline that
            closes the user turn: the information available at newline and all later template tokens
    full    input + the generated answer: the information available at the answer tokens

Each field is a different feature space answering a different question, so it is one baseline row
per (field, condition), never a point on a layer curve. Read a probe against the baseline of the
text it could see: identifier / question / template coordinates against `input`, answer coordinates
against `full`. At identifier coordinates the probe sees only a prefix of the input, so `input` is an
upper bound there.

If TF-IDF already separates the 24 animals under the identifier split, the task is lexically
solvable and a high probe accuracy is not by itself evidence about the representation. If it matches
the probe, compare the error sets (text_baseline_predictions.csv, joinable on uid).

    python text_baseline.py                  # both fields, word + char_wb
    python text_baseline.py --field input
    python text_baseline.py --base results_segments   # default; writes text_baseline.csv,
                                                      # text_baseline_predictions.csv and
                                                      # text_baseline_config.json there
    LEAK_ACTION=skip python text_baseline.py # non-interactive leak handling (see resolve_leaks)

Needs positions.jsonl (answers) in the working directory, as run_segments.py does.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

import data


SEED = 0
FIELDS = ("input", "full")
# word n-grams understate overlap between morphological variants across
# identifiers; char_wb catches those, so both are reported.
ANALYZERS = {"word": (1, 2), "char_wb": (3, 5)}


def build(analyzer, ngram, C=1.0):
    return make_pipeline(
        TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2,
                        sublinear_tf=True),
        LogisticRegression(C=C, max_iter=2000))


def texts(tab, field, ans):
    if field == "input":
        return np.array([r["question"] for r in tab], dtype=object)
    return np.array([r["question"] + " " + ans[r["uid"]] for r in tab], dtype=object)


def find_leaks(classes, txt, tab):
    """Rows whose input contains their own animal's name, grouped by the wording
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
    print(f"\nanimal name appears in its own input: {len(bad)} wording(s), "
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
        raise SystemExit("stopped on leaking input")
    if action in ("i", "ignore"):
        print("continuing WITH the leaking rows -- this inflates the baseline")
        return set()
    drop = {i for i, r in enumerate(tab)
            if (r["animal"], r["ident"], r["wording"]) in bad}
    print(f"skipping {len(drop)} rows (the splits are recomputed on the reduced table and "
          f"no longer match the probes')")
    return drop


def answer_leaks(classes, tab, ans):
    """Share of generated answers containing the animal's own name. Informational: the rows are
    kept, so `full` quantifies how circular the answer coordinates are."""
    hit = np.array([r["animal"].lower() in ans[r["uid"]].lower() for r in tab])
    animal = np.array([r["animal"] for r in tab])
    print(f"\nanimal name in its own generated answer: {hit.mean():.1%} of {len(tab)} answers")
    for c in classes:
        if hit[animal == c].any():
            print(f"  {c:<12} {hit[animal == c].mean():6.1%}")
    return float(hit.mean())


def main(fields=FIELDS, base="results_segments", analyzers=tuple(ANALYZERS), C=1.0):
    out_dir = Path(base)
    out_dir.mkdir(exist_ok=True)
    print(f"-> {out_dir}")

    rows, tab = data.analysis_rows(data.build_table())
    if not len(tab):
        raise SystemExit(f"no cases: data.RUN_DIR={data.RUN_DIR} holds no rows.jsonl / index_*.jsonl "
                         f"-- run with ACTS=<folder that has them>, as for run_segments.py")
    classes = np.unique([r["animal"] for r in tab])

    ans = {}
    if "full" in fields:
        for line in open("positions.jsonl"):
            r = json.loads(line)
            ans[r["uid"]] = r["answer"]
        assert all(r["uid"] in ans for r in tab), "positions.jsonl lacks answers for some cases"

    # An animal name inside its own input would make this trivially circular.
    leak_action = "none"
    bad = find_leaks(classes, texts(tab, "input", ans), tab)
    if bad:
        drop = resolve_leaks(bad, tab)
        leak_action = "skip" if drop else "ignore"
        if drop:
            keep = np.array([i for i in range(len(tab)) if i not in drop])
            rows, tab = rows[keep], [tab[i] for i in keep]
            assert len(np.unique([r["animal"] for r in tab])) == len(classes), \
                "skipping the leaking rows removed a whole class"
    ans_leak = answer_leaks(classes, tab, ans) if "full" in fields else None

    classes, y = np.unique([r["animal"] for r in tab], return_inverse=True)
    uid = np.array([r["uid"] for r in tab])

    (out_dir / "text_baseline_config.json").write_text(json.dumps(
        dict(script="text_baseline", fields=list(fields), analyzers=list(analyzers),
             C=C, run_dir=str(data.RUN_DIR), seed=SEED, n_rows=len(tab),
             leaks=[list(k) for k in sorted(bad)], leak_action=leak_action,
             answer_leak_rate=ans_leak), indent=2))

    conds = data.conditions(tab, seed=SEED)
    n_match = min(len(idx["train"]) for _, _, idx in conds)
    print(f"{len(tab)} rows, {len(classes)} classes, n_train matched to {n_match}")

    with open(out_dir / "text_baseline.csv", "w", newline="") as fh, \
            open(out_dir / "text_baseline_predictions.csv", "w", newline="") as pf:
        w, pw = csv.writer(fh), csv.writer(pf)
        w.writerow(["field", "feature", "condition", "split", "n_train", "n_eval", "acc", "f1"])
        pw.writerow(["field", "feature", "condition", "uid", "y_true", "y_pred"])

        for field in fields:
            txt = texts(tab, field, ans)
            for an in analyzers:
                name_f = f"tfidf-{an}"
                for name, _, idx in conds:
                    tr = data.match_size(idx["train"], y, n_match, seed=SEED)
                    m = build(an, ANALYZERS[an], C)
                    m.fit(txt[tr], y[tr])
                    for part, ev in [("train", tr), ("dev", idx["dev"]),
                                     ("test", idx["test"])]:
                        if len(ev) == 0:                      # no dev split any more
                            continue
                        p = m.predict(txt[ev])
                        w.writerow([field, name_f, name, part, len(tr), len(ev),
                                    f"{accuracy_score(y[ev], p):.4f}",
                                    f"{f1_score(y[ev], p, average='macro'):.4f}"])
                        if part == "test":
                            for u, t, q in zip(uid[ev], y[ev], p):
                                pw.writerow([field, name_f, name, u, classes[t], classes[q]])
                            print(f"{field:<6} {name_f:<14} {name:<10} "
                                  f"test acc {accuracy_score(y[ev], p):.3f}", flush=True)
                    fh.flush()
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="both", choices=["both", *FIELDS])
    ap.add_argument("--analyzer", default="both",
                    choices=["both", "word", "char_wb"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--base", default="results_segments", help="folder of the segment results")
    a = ap.parse_args()
    main(FIELDS if a.field == "both" else (a.field,), a.base,
         tuple(ANALYZERS) if a.analyzer == "both" else (a.analyzer,), a.C)