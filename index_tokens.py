"""Per-token meta info for every case, for any model in models.py.

    python3 index_tokens.py <model> [--out DIR]    # reads DIR/positions.jsonl (index_positions.py) + animals.json

Every token of a case is numbered t = 0 .. n-1 over the sequence Path 2 feeds the model (prompt + answer + closing
token, the input_ids of positions.jsonl). Writes DIR/tokens.parquet: one row per (uid, t) with the token string, its
region, its names and flags. Identical to the Olmo-7B version; only the template pieces and the assistant header
names come from the model's chat template and models.py (cfg.header) instead of being hard-coded.

  regions (exactly one is True):
    is_part_of_system / is_part_of_user_formatting / is_part_of_user_input /
    is_part_of_assistant_formatting (the whole assistant header and the closing token) / is_part_of_assistant_input
  name / name_from_end               the coordinate names run_segments.py probes:
    start_of_ident, intermediate_ident_n, end_of_ident        identifier wording
    after_ident, intermediate_question_n, end_of_question      question text after the identifier
    end_user_formatting, newline, begin_assistant_formatting, <cfg.header names>   template between question and answer
    begin_of_answer, intermediate_answer_n, end_of_answer     generated answer
    end_assistant_formatting                                  closing token
    intermediate tokens are counted from the start in name (_1, _2, ...) and from the end in name_from_end (_-1, ..);
    a one-token segment has the first name in name and the last in name_from_end. "" = not tested: system turn,
    user header, question text before the identifier.
  one boolean column per named role; answer_value = tokens of the answer value inside the answer
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import cases
import models

REGIONS = ["is_part_of_system", "is_part_of_user_formatting", "is_part_of_user_input",
           "is_part_of_assistant_formatting", "is_part_of_assistant_input"]


def named(cfg):
    return ["start_of_ident", "end_of_ident", "after_ident", "end_of_question", "end_user_formatting", "newline",
            "begin_assistant_formatting", *[n for n, _ in cfg.header], "begin_of_answer", "end_of_answer",
            "end_assistant_formatting"]


def tokens_table(rows, tok, cfg, animals):
    before, after, user_close = cases.template_parts(tok, cfg.template_kwargs)
    user_head = before.rindex(cfg.turn_start)                  # start of the user header

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
        head = q_end + len(user_close)                     # start of the assistant header

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
        assert fa + len(cfg.header) == pl, f"assistant header is not {len(cfg.header)} tokens: {r['uid']}"
        for i, nm in [(eu, "end_user_formatting"), (eu + 1, "newline"), (fa, "begin_assistant_formatting"),
                      *[(fa + 1 + k, nm) for k, (nm, _) in enumerate(cfg.header)]]:
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
    for nm in named(cfg):
        meta[nm] = (meta["name"] == nm) | (meta["name_from_end"] == nm)

    # checks: the template roles sit on the tokens they should, in every case
    close = meta.loc[meta["end_user_formatting"], "token"].unique()
    want = [("end_user_formatting", list(close)), ("newline", ["\n"]), ("begin_assistant_formatting", [cfg.turn_start]),
            *[(nm, [t]) for nm, t in cfg.header], ("end_assistant_formatting", list(close))]
    for nm, w in want:
        got = meta.loc[meta[nm], "token"].unique()
        assert list(got) == w and len(close) == 1, f"{nm} sits on {got[:5]}"
    stats = dict(cases=len(rows), tokens=len(meta), answer_value_found=found / len(rows),
                 correct_but_not_found=correct_unfound, tokens_across_region_border=straddle)
    return meta, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=models.MODELS)
    ap.add_argument("--out", default=".")
    ap.add_argument("--data", default="animals.json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    cfg, out = models.MODELS[a.model], Path(a.out)
    rows = [json.loads(line) for line in open(out / "positions.jsonl")]
    meta, stats = tokens_table(rows, AutoTokenizer.from_pretrained(cfg.id), cfg, cases.load_animals(a.data))
    meta.to_parquet(out / "tokens.parquet", index=False)
    print(stats, f"-> {out / 'tokens.parquet'}")
