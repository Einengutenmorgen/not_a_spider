"""Per-token meta info for every case.

    python3 index_tokens.py            # reads positions.jsonl (index_positions.py) + animals.json

Every token of a case is numbered t = 0 .. n-1 over the sequence Path 2 feeds the model
(prompt + answer + closing <|im_end|>, the input_ids in positions.jsonl).

tokens.parquet   one row per (uid, t): the token string, its region, its names and flags.
                 meta[meta.uid == case_id] is the per-case dict t -> {...}.
  regions (exactly one is True):
    is_part_of_system                 system turn added by the chat template
    is_part_of_user_formatting        <|im_start|>user\\n  and  <|im_end|>\\n after the question
    is_part_of_user_input             the question text
    is_part_of_assistant_formatting   <|im_start|>assistant\\n  and the closing <|im_end|>
    is_part_of_assistant_input        the answer: the whole generated text
  name / name_from_end               the coordinate names run_segments.py probes:
    start_of_ident, intermediate_ident_n, end_of_ident        identifier wording
    after_ident, intermediate_question_n, end_of_question     question text after the identifier
    end_user_formatting, newline, begin_assistant_formatting, assistant_role,
    assistant_header_newline                                   template between question and answer
    begin_of_answer, intermediate_answer_n, end_of_answer     generated answer
    end_assistant_formatting                                   closing <|im_end|>
    intermediate tokens are counted from the start in name (_1, _2, ...) and from the end in
    name_from_end (_-1, _-2, ...); a one-token segment has the first name in name and the last
    in name_from_end. "" = not tested: system turn, user header, question text before the
    identifier.
  one boolean column per named role (start_of_ident, ..., end_assistant_formatting): True where
    name or name_from_end is that role
  answer_value                        tokens of the answer value inside the answer: the first term
                                      from the slot's accepted answers (any animal), or the first
                                      number for range slots, e.g. "4" or "four"
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

MODEL = os.environ.get("MODEL", "allenai/Olmo-3-7B-Instruct")
REGIONS = ["is_part_of_system", "is_part_of_user_formatting", "is_part_of_user_input",
           "is_part_of_assistant_formatting", "is_part_of_assistant_input"]
NAMED = ["start_of_ident", "end_of_ident", "after_ident", "end_of_question", "end_user_formatting",
         "newline", "begin_assistant_formatting", "assistant_role", "assistant_header_newline",
         "begin_of_answer", "end_of_answer", "end_assistant_formatting"]

tok = AutoTokenizer.from_pretrained(MODEL)
animals = json.loads(Path("animals.json").read_text())
rows = [json.loads(line) for line in open("positions.jsonl")]

# template pieces around the question, as in index_positions.py
SENTINEL = "\x00QUESTION\x00"
msgs = [{"role": "user", "content": SENTINEL}]
before, after = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True).split(SENTINEL)
user_close = tok.apply_chat_template(msgs, tokenize=False).split(SENTINEL)[1]
user_head = before.rindex("<|im_start|>")              # start of "<|im_start|>user\n"

# answer value vocabulary per slot: accepted terms of every animal (catches wrong answers too)
vocab, ranges = {}, set()
for a in animals.values():
    for slot, spec in a["attributes"].items():
        if "accepted" in spec:
            vocab.setdefault(slot, set()).update(spec["accepted"])
        elif "min" in spec:
            ranges.add(slot)
value_re = {slot: re.compile("|".join(r"\b" + re.escape(v) + r"\b" for v in sorted(vs, key=len, reverse=True)), re.I)
            for slot, vs in vocab.items()}
number_re = re.compile(r"\d+(?:\.\d+)?")


def answer_span(slot, text):
    m = (value_re[slot] if slot in value_re else number_re if slot in ranges else None)
    m = m.search(text) if m else None
    return (m.start(), m.end()) if m else None


cols = {k: [] for k in ["uid", "t", "token", *REGIONS, "name", "name_from_end", "answer_value"]}
found, correct_unfound, straddle = 0, 0, 0
for r in rows:
    animal, slot, ident, w, f = r["uid"].split("|")
    frame = animals[animal]["question_frames"][slot][int(f)]
    wording = animals[animal]["identifiers"][ident]["wordings"][int(w)]
    q = frame.replace("{ID}", wording)
    prompt = before + q + after
    q_start, q_end = len(before), len(before) + len(q)
    head = q_end + len(user_close)                     # start of "<|im_start|>assistant\n"

    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ans = tok(r["answer"], add_special_tokens=False, return_offsets_mapping=True)
    n_p, n_a = len(enc.input_ids), len(ans.input_ids)
    assert enc.input_ids + ans.input_ids == r["input_ids"][:-1], f"tokens differ from positions.jsonl: {r['uid']}"

    starts = [a for a, _ in enc.offset_mapping]
    straddle += sum(a < c < b for a, b in enc.offset_mapping for c in (user_head, q_start, q_end, head))
    region = np.searchsorted([user_head, q_start, q_end, head], starts, side="right")  # 0..4
    region = list(region) + [5] * n_a + [6]                                          # answer, close
    span = answer_span(slot, r["answer"])
    is_answer = [False] * n_p + [span is not None and a < span[1] and b > span[0]
                                 for a, b in ans.offset_mapping] + [False]
    found += span is not None
    correct_unfound += bool(r.get("correct")) and span is None

    n = n_p + n_a + 1
    name, name_end = [""] * n, [""] * n                 # "" = not tested

    def seg(first, middle, last, a, b):                 # tokens a..b, inclusive
        if b < a:
            return
        name[a], name_end[a], name[b], name_end[b] = first, first, last, last
        name[a] = first                                 # one token: first in name, last in name_end
        for k in range(1, b - a):
            name[a + k], name_end[b - k] = f"{middle}_{k}", f"{middle}_-{k}"

    eu, fa, pl = r["end_user_formatting"], r["first_assistant_temp"], r["prompt_last"]
    seg("start_of_ident", "intermediate_ident", "end_of_ident", r["start_of_ident"], r["end_of_ident"])
    seg("after_ident", "intermediate_question", "end_of_question", r["end_of_ident"] + 1, eu - 1)
    for i, nm in [(eu, "end_user_formatting"), (eu + 1, "newline"), (fa, "begin_assistant_formatting"),
                  *[(i, "assistant_role" if pl - fa == 2 else f"assistant_role_{i - fa - 1}")
                    for i in range(fa + 1, pl)], (pl, "assistant_header_newline")]:
        name[i] = name_end[i] = nm
    seg("begin_of_answer", "intermediate_answer", "end_of_answer", n_p, n_p + n_a - 1)
    name[n - 1] = name_end[n - 1] = "end_assistant_formatting"
    assert all(name[r["start_of_ident"]:]), f"unnamed token after the identifier start: {r['uid']}"

    cols["uid"] += [r["uid"]] * n
    cols["t"] += range(n)
    cols["token"] += r["input_ids"]
    # regions: 0 system, 1 user header, 2 question, 3 user close, 4 assistant header, 5 answer, 6 close
    for col, ids_ in zip(REGIONS, [(0,), (1, 3), (2,), (4, 6), (5,)]):
        cols[col] += [g in ids_ for g in region]
    cols["name"] += name
    cols["name_from_end"] += name_end
    cols["answer_value"] += is_answer

meta = pd.DataFrame(cols)
meta["token"] = meta["token"].map({i: tok.decode([i]) for i in meta["token"].unique()})   # " the", "\n"
assert (meta[REGIONS].sum(axis=1) == 1).all(), "a token has no or several regions"
for nm in NAMED:
    meta[nm] = (meta["name"] == nm) | (meta["name_from_end"] == nm)
meta.to_parquet("tokens.parquet", index=False)

# checks: the template roles sit on the tokens they should, in every case
for nm, want in [("end_user_formatting", "<|im_end|>"), ("newline", "\n"), ("begin_assistant_formatting", "<|im_start|>"),
                 ("assistant_header_newline", "\n"), ("end_assistant_formatting", "<|im_end|>")]:
    got = meta.loc[meta[nm], "token"].unique()
    assert list(got) == [want], f"{nm} sits on {got[:5]}"
print(f"{len(rows)} cases, {len(meta)} tokens | answer value found in {found / len(rows):.1%} of cases"
      f" | correct but not found: {correct_unfound} | tokens across a region border: {straddle}")
