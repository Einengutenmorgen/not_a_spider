"""Lexical overlap of identifier wordings: within-ident, between-ident (same animal), cross-animal.

CLI (unchanged in spirit):
    python check_overlap.py data/v4/animals.json [old_animals.json]

Importable:
    gate_and_pairs(data)               -> (gate, {(animal, identA, identB): max_cosine})
    check_candidates(data, animal, candidates, exclude=...) -> violation report
"""
import json, re, itertools, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS


HEADS = {"animal", "creature", "species", "beast", "critter", "one", "swimmer",
         "thing", "organism", "paddler"}


def content(s):
    return {w for w in re.findall(r"[a-z]+", s.lower())
            if w not in ENGLISH_STOP_WORDS and w not in HEADS and len(w) > 2}


def build_rows(new, ref=None):
    """(animal, ident, text) for every wording. ref = old-format animals.json, cross-animal only."""
    rows = []
    for animal, a in new.items():
        for k, v in a["identifiers"].items():
            rows += [(animal, k, w) for w in v["wordings"]]
    for a in (ref or []):
        if a["animal"] not in new:
            rows += [(a["animal"], f"old_{i}", w)
                     for i, w in enumerate(a["identifier_variants"])]
    return rows


def similarity(rows):
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                        sublinear_tf=True).fit_transform([r[2] for r in rows])
    return (X @ X.T).toarray()


def analyse(rows, C, in_scope):
    """Bucket every pair; return buckets, between-ident block-max, cross-animal block-max."""
    sets = [content(r[2]) for r in rows]
    buckets = {"within_ident": [], "between_ident": [], "cross_animal": []}
    pair_max, cross_max = {}, {}

    for i, j in itertools.combinations(range(len(rows)), 2):
        (ai, ki, _), (aj, kj, _) = rows[i], rows[j]
        if ai not in in_scope and aj not in in_scope:
            continue

        jac = len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j]))
        b = "cross_animal" if ai != aj else ("within_ident" if ki == kj else "between_ident")
        buckets[b].append((C[i, j], jac))

        if b == "between_ident":
            key = (ai, *sorted((ki, kj)))
            pair_max[key] = max(pair_max.get(key, 0), C[i, j])
        elif b == "cross_animal":
            key = (ai, ki, aj) if ai in in_scope else (aj, kj, ai)
            cross_max[key] = max(cross_max.get(key, 0), C[i, j])

    return buckets, pair_max, cross_max


def gate_and_pairs(data, ref=None):
    """Gate = p95 of cross-animal block-max cosine (same block size as ident pairs)."""
    rows = build_rows(data, ref)
    C = similarity(rows)
    buckets, pair_max, cross_max = analyse(rows, C, set(data))
    gate = float(np.percentile(list(cross_max.values()), 95)) if cross_max else 1.0
    return gate, pair_max, buckets


def shared_content_words(animal_entry):
    """Content words used by more than one identifier of the same animal."""
    owner = {}
    for k, v in animal_entry["identifiers"].items():
        for w in set().union(*map(content, v["wordings"])):
            owner.setdefault(w, set()).add(k)
    return {w: sorted(s) for w, s in owner.items() if len(s) > 1}


def check_candidates(data, animal, candidates, exclude=(), gate=None, ref=None):
    """Are `candidates` conceptually new for `animal`?

    candidates: {provisional_ident_key: [wordings]}
    exclude:    ident keys of `animal` being replaced (not compared against)

    Returns {"gate": float, "ok": [keys], "violations": {key: [(other_key, cosine)]}}
    """
    probe = json.loads(json.dumps(data))  # cheap deep copy
    for k in exclude:
        probe[animal]["identifiers"].pop(k, None)
    for k, wordings in candidates.items():
        probe[animal]["identifiers"][k] = {"features": [], "wordings": list(wordings)}

    g, pair_max, _ = gate_and_pairs(probe, ref)
    gate = g if gate is None else gate

    violations = {k: [] for k in candidates}
    for (a, ka, kb), v in pair_max.items():
        if a != animal or v <= gate:
            continue
        for k in (ka, kb):
            if k in violations:
                violations[k].append((kb if k == ka else ka, round(float(v), 3)))

    return {
        "gate": round(gate, 3),
        "ok": [k for k, v in violations.items() if not v],
        "violations": {k: v for k, v in violations.items() if v},
    }


def main():
    new = json.load(open(sys.argv[1]))
    ref = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None

    rows = build_rows(new, ref)
    C = similarity(rows)
    buckets, pair_max, cross_max = analyse(rows, C, set(new))

    p95 = np.percentile(list(cross_max.values()), 95)
    print(f"{'bucket':14} {'n':>6} {'cos mean':>9} {'cos p95':>8} {'jac mean':>9}")
    for b, v in buckets.items():
        if not v:
            continue
        c, j = np.array(v).T
        print(f"{b:14} {len(v):6d} {c.mean():9.3f} {np.percentile(c,95):8.3f} {j.mean():9.3f}")

    print(f"\ngate: p95 of cross-animal block-max cosine = {p95:.3f}")
    bad = {k: v for k, v in pair_max.items() if v > p95}
    print(f"ident pairs whose max cosine exceeds gate: {len(bad)}/{len(pair_max)}")
    for k, v in sorted(bad.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v:.3f}")

    for animal, a in new.items():
        shared = shared_content_words(a)
        print(f"\n{animal}: content words used by >1 ident ({len(shared)}):")
        for w, s in sorted(shared.items(), key=lambda x: -len(x[1])):
            print(f"  {w}: {', '.join(s)}")

    vocab = {n: set().union(*(content(w) for v in a["identifiers"].values()
                              for w in v["wordings"])) for n, a in new.items()}
    print("\ncross-animal shared content words (note only):")
    for x, y in itertools.combinations(vocab, 2):
        shared = sorted(vocab[x] & vocab[y])
        print(f"  {x}-{y} ({len(shared)}): {', '.join(shared)}")


if __name__ == "__main__":
    main()