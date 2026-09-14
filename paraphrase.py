"""Step 1: per animal, generate N identifier paraphrases + name-free question templates.
Step 2 (--compose): expand templates x identifiers into attribute_questions_paraphrased.

Usage:
    python paraphrase.py animals.json
    python paraphrase.py animals.json --compose
"""
import json, os, re, sys, pathlib
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

N = 10
MODEL = os.environ.get("MODEL", "anthropic/claude-sonnet-5")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

PROMPT = """You rewrite referring expressions for an animal-identification benchmark.

Animal features (this is the ONLY identifying information you may use):
unique features: {unique}
shared features: {shared}
description: {desc}

Attribute questions:
{questions}

Task A. Write exactly {n} different referring expressions for this animal.
- Each is a noun phrase starting with "the", usable inside a sentence.
- They must all identify the same animal with the SAME information as above.
  Do not add identifying facts that are not listed, and do not drop so much that
  the expression no longer picks out one animal.
- Never write the animal's name, a synonym of it, or its genus/species/taxon name.
- Vary wording, syntax and which of the listed features you foreground.

Task B. Return each attribute question VERBATIM, except that the referring
expression for the animal is replaced by the placeholder {{ID}}. Change nothing
else - not the attribute asked about, not the answer-format instruction.

Return only JSON:
{{"identifiers": ["the ...", ...{n} items],
  "question_templates": {{"<question key>": "... {{ID}} ...", ...all keys}}}}"""


def entries(data):
    return data if isinstance(data, list) else data["animals"]


def save(data, path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def leaks(text, entry):
    names = [entry["animal"], *entry.get("accepted_answers", [])]
    return [n for n in names if re.search(rf"\b{re.escape(n)}\b", text, re.I)]


def ask(entry, extra=""):
    q = entry["attribute_questions"]
    msg = PROMPT.format(
        n=N,
        unique="; ".join(entry.get("unique_features", [])),
        shared="; ".join(f for grp in entry.get("shared_features", []) for f in grp),
        desc=entry.get("description", ""),
        questions="\n".join(f"{k}: {v}" for k, v in q.items()),
    ) + extra
    raw = client.chat.completions.create(
        model=MODEL, max_tokens=4000, messages=[{"role": "user", "content": msg}]
    ).choices[0].message.content
    return json.loads(re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M).strip())


def validate(out, entry):
    ids, tpl = out["identifiers"], out["question_templates"]
    if len(ids) != N:
        return f"Expected {N} identifiers, got {len(ids)}."
    if set(tpl) != set(entry["attribute_questions"]):
        return "question_templates keys must match the attribute question keys exactly."
    for t in tpl.values():
        if t.count("{ID}") != 1:
            return f"Each template needs exactly one {{ID}}: {t!r}"
    for text in [*ids, *tpl.values()]:
        bad = leaks(text, entry)
        if bad:
            return f"{text!r} contains the forbidden name(s) {bad}."
    return None


def step1(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    bar = tqdm(entries(data), unit="animal")
    for entry in bar:
        bar.set_description(entry["animal"][:20])
        if entry.get("identifier_variants"):
            continue
        extra = ""
        for attempt in range(3):
            try:
                out = ask(entry, extra)
                err = validate(out, entry)
            except Exception as e:  # bad JSON, API hiccup
                out, err = None, f"Previous reply failed: {e}"
            if err is None:
                break
            tqdm.write(f"  retry {entry['animal']}: {err}")
            extra = f"\n\nYour previous answer was rejected: {err} Fix it."
        else:
            tqdm.write(f"SKIPPED {entry['animal']}")
            continue
        entry["identifier_variants"] = out["identifiers"]
        entry["question_templates"] = out["question_templates"]
        save(data, path)


def compose(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in tqdm(entries(data), unit="animal"):
        ids, tpl = entry.get("identifier_variants"), entry.get("question_templates")
        if not ids or not tpl:
            continue
        entry["attribute_questions_paraphrased"] = {
            k: [t.replace("{ID}", i) for i in ids] for k, t in tpl.items()
        }
    save(data, path)


if __name__ == "__main__":
    p = pathlib.Path(sys.argv[1])
    compose(p) if "--compose" in sys.argv else step1(p)