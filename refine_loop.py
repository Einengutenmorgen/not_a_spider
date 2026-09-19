"""Iteratively repair identifiers until every one reaches accept status.

Per iteration:
  1. read data/v{n}/animals.json + data/v{n}/jeopardy_check.json
  2. per animal, ask the LLM for (a) replacements for rejected identifiers,
     (b) replacement wordings for the failed wordings of accepted identifiers
  3. gate candidates with check_overlap (concept must be new, not a reword)
  4. write data/v{n+1}/animals.json
  5. re-run check_jeopardy on it
  stop when no identifier is rejected and no wording failed.

python refine_loop.py [--start v3] [--max-iters 5] [--animals Walrus Chameleon]
"""
import argparse, json, os, re, sys, time
import requests

import check_jeopardy as cj
import check_overlap as co
from dotenv import load_dotenv

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openai/gpt-5.6-luna"
HISTORY_PATH = os.path.join("data", "refine_history.json")


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------




def version_dirs():
    if not os.path.isdir("data"):
        return []
    vs = [(int(m.group(1)), d) for d in os.listdir("data")
          for m in [re.fullmatch(r"v(\d+)", d)] if m]
    return [d for _, d in sorted(vs)]


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------
# history of everything already declined
# ----------------------------------------------------------------------

def load_history():
    return read_json(HISTORY_PATH, {"declined_identifiers": {}, "declined_wordings": {}})


def record_declined_identifier(hist, animal, entry, reason):
    hist["declined_identifiers"].setdefault(animal, []).append(
        {"features": entry.get("features", []),
         "wordings": entry.get("wordings", []),
         "reason": reason})


def record_declined_wordings(hist, animal, wordings, reason):
    bucket = hist["declined_wordings"].setdefault(animal, [])
    known = {w["text"] for w in bucket}
    for w in wordings:
        if w not in known:
            bucket.append({"text": w, "reason": reason})
            known.add(w)


def declined_identifier_texts(hist, animal):
    return [e["features"] and e["features"][0] or (e["wordings"][:1] or [""])[0]
            for e in hist["declined_identifiers"].get(animal, [])]


def declined_wording_texts(hist, animal):
    return [w["text"] for w in hist["declined_wordings"].get(animal, [])]


# ----------------------------------------------------------------------
# what needs fixing
# ----------------------------------------------------------------------

def todo_from_summary(summary):
    """-> {animal: {"rejected": [ident], "failed": {ident: [wording_idx]}}}"""
    todo = {}
    for animal, idents in summary.items():
        rejected, failed = [], {}
        for ident, s in idents.items():
            if s["status"] == "reject":
                rejected.append(ident)
            else:
                bad = sorted(set(s.get("frame0_failed", [])) | set(s.get("frame1_failed", [])))
                if bad:
                    failed[ident] = bad
        if rejected or failed:
            todo[animal] = {"rejected": rejected, "failed": failed}
    return todo


# ----------------------------------------------------------------------
# OpenRouter
# ----------------------------------------------------------------------

def ask_llm(messages, model=LLM_MODEL, retries=3):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY missing (.env or environment)")

    body = {"model": model, "messages": messages,
            "response_format": {"type": "json_object"}}

    for attempt in range(retries):
        r = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json=body, timeout=180)
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"]
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                if attempt == retries - 1:
                    raise
        else:
            print(f"  openrouter {r.status_code}: {r.text[:200]}")
            if attempt == retries - 1:
                r.raise_for_status()
        time.sleep(2 * (attempt + 1))


RULES = """Hard rules for every wording you produce:
- The animal's name (and any of its alternative names) must NEVER appear.
- A wording is a noun phrase that stands in for the animal, e.g. "the colour-changing lizard kept as a pet".
- It must be resolvable: a competent reader, and a 7B instruct model, must be able to name the animal from it alone.
- All wordings of one identifier express the SAME underlying feature, only phrased differently.
- Different identifiers of the same animal must express DIFFERENT concepts, not rephrasings of each other.
- Avoid the listed failed/declined phrasings and the concepts already in use.
- Prefer widely known, unambiguous facts over niche taxonomy or a single species' trait."""


def prompt_new_identifiers(data, animal, idents, summary, hist, feedback=None):
    a = data[animal]
    existing = {k: v["features"] for k, v in a["identifiers"].items() if k not in idents}

    failing = {}
    for k in idents:
        s = summary[animal][k]
        failing[k] = {
            "features": a["identifiers"][k]["features"],
            "wordings": a["identifiers"][k]["wordings"],
            "frame0_rate": round(s["frame0_rate"], 3),
            "frame1_rate": round(s["frame1_rate"], 3),
        }

    user = {
        "animal": animal,
        "alternative_names_forbidden": a["alternative_naming"],
        "attributes_asked_later": list(a["attributes"]),
        "identifiers_to_replace": failing,
        "concepts_already_in_use": existing,
        "previously_declined_concepts": declined_identifier_texts(hist, animal),
        "previously_declined_wordings": declined_wording_texts(hist, animal),
        "task": (
            f"Replace each identifier in identifiers_to_replace with a NEW identifier "
            f"expressing a different, more recognisable feature of the {animal}. "
            f"Give exactly 10 wordings per identifier. Also list which of "
            f"attributes_asked_later the new feature gives away (leaks)."
        ),
        "output_schema": {
            "<ident_key>": {"features": ["one short feature description"],
                            "wordings": ["10 strings"],
                            "leaks": ["attribute names or empty"]}
        },
    }
    if feedback:
        user["rejected_last_attempt"] = feedback

    return [{"role": "system", "content": RULES + "\nAnswer with JSON only."},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)}]


def prompt_new_wordings(data, animal, failed, hist, feedback=None):
    a = data[animal]
    items = {}
    for ident, idx in failed.items():
        v = a["identifiers"][ident]
        items[ident] = {
            "features": v["features"],
            "keep_these_wordings": [w for i, w in enumerate(v["wordings"]) if i not in idx],
            "replace_these_wordings": [v["wordings"][i] for i in idx],
            "n_needed": len(idx),
        }

    user = {
        "animal": animal,
        "alternative_names_forbidden": a["alternative_naming"],
        "identifiers": items,
        "other_concepts_in_use": {k: v["features"] for k, v in a["identifiers"].items()
                                  if k not in failed},
        "previously_declined_wordings": declined_wording_texts(hist, animal),
        "task": ("For each identifier, produce exactly n_needed NEW wordings of the SAME "
                 "feature. They replace replace_these_wordings, which the target model "
                 "failed to resolve. Do not repeat keep_these_wordings or any declined one."),
        "output_schema": {"<ident_key>": ["n_needed strings"]},
    }
    if feedback:
        user["rejected_last_attempt"] = feedback

    return [{"role": "system", "content": RULES + "\nAnswer with JSON only."},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)}]


# ----------------------------------------------------------------------
# validation of candidates
# ----------------------------------------------------------------------

def leaks_name(wordings, names):
    return [w for w in wordings if cj.hit(w, names)]


def validate_identifiers(data, animal, cand, replacing, hist):
    """Name leak + overlap gate. Returns (accepted, feedback)."""
    names = data[animal]["alternative_naming"]
    feedback, clean = {}, {}

    for k, v in cand.items():
        wordings = v.get("wordings", [])
        if len(wordings) < 10:
            feedback[k] = f"need 10 wordings, got {len(wordings)}"
            continue
        leaked = leaks_name(wordings, names)
        if leaked:
            feedback[k] = f"wordings contain the animal name: {leaked}"
            record_declined_wordings(hist, animal, leaked, "name leak")
            continue
        clean[k] = v

    if clean:
        report = co.check_candidates(
            data, animal, {k: v["wordings"] for k, v in clean.items()},
            exclude=replacing)
        for k, hits in report["violations"].items():
            feedback[k] = (f"too similar to existing identifier(s) "
                           f"{[h[0] for h in hits]} (cosine {[h[1] for h in hits]} "
                           f"> gate {report['gate']}) — needs a different concept")
            record_declined_identifier(hist, animal, clean[k], "overlap gate")
        clean = {k: v for k, v in clean.items() if k in report["ok"]}

    return clean, feedback


def validate_wordings(data, animal, cand, failed, hist):
    names = data[animal]["alternative_naming"]
    feedback, clean = {}, {}

    for k, wordings in cand.items():
        need = len(failed.get(k, []))
        wordings = [w for w in wordings if isinstance(w, str)]
        if len(wordings) < need:
            feedback[k] = f"need {need} wordings, got {len(wordings)}"
            continue
        leaked = leaks_name(wordings, names)
        if leaked:
            feedback[k] = f"wordings contain the animal name: {leaked}"
            record_declined_wordings(hist, animal, leaked, "name leak")
            continue
        clean[k] = wordings[:need]

    if clean:
        # a replacement wording must not drag its identifier into another one's concept
        probe = {k: [w for i, w in enumerate(data[animal]["identifiers"][k]["wordings"])
                     if i not in failed[k]] + v for k, v in clean.items()}
        report = co.check_candidates(data, animal, probe, exclude=list(clean))
        for k, hits in report["violations"].items():
            feedback[k] = (f"new wordings collide with {[h[0] for h in hits]} "
                           f"(cosine {[h[1] for h in hits]} > gate {report['gate']})")
            record_declined_wordings(hist, animal, clean[k], "overlap gate")
        clean = {k: v for k, v in clean.items() if k in report["ok"]}

    return clean, feedback


# ----------------------------------------------------------------------
# one refinement pass over the data
# ----------------------------------------------------------------------

def refine(data, summary, hist, llm_model, max_tries=3):
    todo = todo_from_summary(summary)
    new = json.loads(json.dumps(data))
    changed = 0

    for animal, work in todo.items():
        # --- rejected identifiers: replace the whole concept -------------
        if work["rejected"]:
            pending, feedback = list(work["rejected"]), None
            for attempt in range(max_tries):
                if not pending:
                    break
                print(f"  {animal}: new identifiers for {pending} (try {attempt + 1})")
                cand = ask_llm(prompt_new_identifiers(
                    data, animal, pending, summary, hist, feedback), llm_model) or {}
                cand = {k: v for k, v in cand.items() if k in pending}

                good, feedback = validate_identifiers(data, animal, cand, pending, hist)
                for k, v in good.items():
                    record_declined_identifier(
                        hist, animal, data[animal]["identifiers"][k], "rejected by check")
                    new[animal]["identifiers"][k] = {
                        "features": v["features"], "wordings": v["wordings"][:10]}
                    leaks = [x for x in v.get("leaks", []) if x in data[animal]["attributes"]]
                    if leaks:
                        new[animal].setdefault("leakage", {})[k] = leaks
                    else:
                        new[animal].get("leakage", {}).pop(k, None)
                    changed += 1
                pending = [k for k in pending if k not in good]
            if pending:
                print(f"  {animal}: gave up on {pending} this iteration")

        # --- accepted identifiers with individual failed wordings --------
        if work["failed"]:
            pending, feedback = dict(work["failed"]), None
            for attempt in range(max_tries):
                if not pending:
                    break
                print(f"  {animal}: new wordings for "
                      f"{ {k: len(v) for k, v in pending.items()} } (try {attempt + 1})")
                cand = ask_llm(prompt_new_wordings(
                    data, animal, pending, hist, feedback), llm_model) or {}
                cand = {k: v for k, v in cand.items() if k in pending}

                good, feedback = validate_wordings(data, animal, cand, pending, hist)
                for k, repl in good.items():
                    ws = list(new[animal]["identifiers"][k]["wordings"])
                    record_declined_wordings(
                        hist, animal, [ws[i] for i in pending[k]], "failed to resolve")
                    for i, w in zip(pending[k], repl):
                        ws[i] = w
                    new[animal]["identifiers"][k]["wordings"] = ws
                    changed += 1
                pending = {k: v for k, v in pending.items() if k not in good}
            if pending:
                print(f"  {animal}: gave up on wordings {list(pending)} this iteration")

    return new, changed


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="version dir to start from, e.g. v3 (default: latest)")
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--animals", nargs="*")
    ap.add_argument("--llm-model", default=LLM_MODEL)
    ap.add_argument("--target-model", default=cj.DEFAULT_MODEL)
    ap.add_argument("--min-rate", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-tries", type=int, default=3, help="LLM retries per animal per stage")
    ap.add_argument("--gpu", default="1")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    load_dotenv()

    versions = version_dirs()
    start = args.start or (versions[-1] if versions else sys.exit("no data/v* found"))
    n = int(start[1:])

    data = read_json(f"data/v{n}/animals.json") or sys.exit(f"missing data/v{n}/animals.json")
    animals = args.animals or list(data)
    hist = load_history()

    print(f"loading target model {args.target_model} ...")
    model, tok = cj.load_model(args.target_model)

    check = read_json(f"data/v{n}/jeopardy_check.json")
    if check and check.get("min_rate") == args.min_rate:
        summary = {a: s for a, s in check["summary"].items() if a in animals}
        print(f"reusing data/v{n}/jeopardy_check.json")
    else:
        print(f"no usable check for v{n} — running it")
        summary, records = cj.run_check(data, animals, model, tok, args.min_rate,
                                        args.batch_size, args.max_new_tokens)
        cj.write_reports(f"data/v{n}/jeopardy_check.json", summary, records,
                         args.target_model, f"data/v{n}/animals.json", args.min_rate)

    for it in range(args.max_iters):
        todo = todo_from_summary(summary)
        if not todo:
            print(f"\nall identifiers accept with no failed wordings at v{n} — done")
            break

        print(f"\n=== iteration {it + 1}: v{n} -> v{n + 1} ===")
        print(f"animals needing work: {len(todo)} "
              f"({sum(len(w['rejected']) for w in todo.values())} rejected idents, "
              f"{sum(len(w['failed']) for w in todo.values())} idents with failed wordings)")

        data, changed = refine(data, summary, hist, args.llm_model, args.max_tries)
        write_json(HISTORY_PATH, hist)

        if not changed:
            print("nothing could be replaced this iteration — stopping")
            break

        n += 1
        write_json(f"data/v{n}/animals.json", data)
        print(f"wrote data/v{n}/animals.json ({changed} identifiers/wording sets changed)")

        summary, records = cj.run_check(data, animals, model, tok, args.min_rate,
                                        args.batch_size, args.max_new_tokens)
        cj.write_reports(f"data/v{n}/jeopardy_check.json", summary, records,
                         args.target_model, f"data/v{n}/animals.json", args.min_rate)
    else:
        print(f"\nstopped after {args.max_iters} iterations; latest is v{n}")

    write_json(HISTORY_PATH, hist)


if __name__ == "__main__":
    main()
