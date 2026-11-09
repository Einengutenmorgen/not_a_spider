"""Prompt build and scoring, importable. Same cases, same order, same scoring as extract_acts_sharded.py.

Canonical row order = the build loop of the Olmo-7B run: animal -> identifier -> slot (leaking pairs skipped)
-> wording -> frame. It does not depend on the model, so every model's table can be put in this order.
"""
import json
import re
from pathlib import Path

SENTINEL = "\x00QUESTION\x00"


def load_animals(path="animals.json"):
    return json.loads(Path(path).read_text())


def template_parts(tok, kw=None):
    """The chat template is rendered once around a sentinel: prompt = before + question + after.
    user_close = what the template writes after the question when no generation prompt is added."""
    kw, msgs = kw or {}, [{"role": "user", "content": SENTINEL}]
    before, after = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw).split(SENTINEL)
    user_close = tok.apply_chat_template(msgs, tokenize=False, **kw).split(SENTINEL)[1]
    assert after.startswith(user_close), "generation prompt does not extend the user turn"
    return before, after, user_close


def canonical_rows(animals):
    """The cases in canonical order, without prompts (no tokenizer needed). This is the row order of every model's
    tables: run_segments and the analyses take it from here, not from the order of the index files."""
    rows = []
    for animal, a in animals.items():
        for ident_key, ident in a["identifiers"].items():
            for slot in a["attributes"]:
                if slot in a["leakage"].get(ident_key, []):
                    continue                                    # this concept gives the attribute away
                for wi, wording in enumerate(ident["wordings"]):
                    for fi, frame in enumerate(a["question_frames"][slot]):
                        rows.append(dict(uid=f"{animal}|{slot}|{ident_key}|{wi}|{fi}", animal=animal, slot=slot,
                                         ident=ident_key, wording=wi, frame=fi,
                                         question=frame.replace("{ID}", wording)))
    return rows


def build_cases(animals, tok, kw=None):
    kw = kw or {}
    before, after, _ = template_parts(tok, kw)
    cases = canonical_rows(animals)
    for c in cases:
        c["full"] = before + c["question"] + after
    for s in range(0, len(cases), 20000):
        chunk = cases[s:s + 20000]
        for c, ids in zip(chunk, tok([c["full"] for c in chunk], add_special_tokens=False).input_ids):
            c["input_ids"], c["n_tokens"] = ids, len(ids)
    for c in cases:
        c.update(answer="", correct=False, reliable=True)
    # checks of the original script, plus: the sentinel build equals the template applied to the real question
    assert len({c["uid"] for c in cases}) == len(cases), "uid collision"
    assert len({c["question"] for c in cases}) == len(cases), "duplicate question text"
    assert len({c["input_ids"][-1] for c in cases}) == 1, "last prompt token differs across cases"
    for c in cases[::max(1, len(cases) // 200)]:
        ref = tok.apply_chat_template([{"role": "user", "content": c["question"]}], add_generation_prompt=True,
                                      return_dict=True, **kw)["input_ids"]
        assert list(ref) == c["input_ids"], f"tokenization differs from the chat template: {c['uid']}"
    return cases


def normalize(text):
    return ' '.join(re.sub(r'[^\w\s]', ' ', text.lower()).split())


def hit(candidates, answer):
    """Word-boundary search: 'bat' will not match 'wombat', and 'Four.' matches 'four'."""
    n = normalize(answer)
    return any(re.search(rf'\b{re.escape(normalize(c))}\b', n) for c in candidates)


def score(animals, c):
    """Lenient on purpose, as in the Olmo-7B run: any accepted term anywhere in the answer; numeric slots: first number."""
    spec = animals[c["animal"]]["attributes"][c["slot"]]
    if spec["type"] == "set":
        return hit(spec["accepted"], c["answer"])
    m = re.search(r'\d+(?:\.\d+)?', c["answer"])               # unit is NOT checked
    return bool(m) and spec["min"] <= float(m.group()) <= spec["max"]
