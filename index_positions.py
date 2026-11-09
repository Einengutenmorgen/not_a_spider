"""Anchor token positions for every case, for any model in models.py.

    python3 index_positions.py <model> acts/<RID> [--out DIR]   # reads index_*.jsonl; writes DIR/positions.jsonl

Same rows and anchors as the Olmo-7B script (OLD_index_positions.py): input_ids = chat-templated prompt + the
re-encoded answer + the token that closes the user turn (it closes the assistant turn too); Path 2 feeds exactly
these. The template comes from the model's own chat template (cases.template_parts), the assistant header from
models.py; the layout is checked, never assumed:

    start_of_ident / end_of_ident   first / last token of the identifier wording
    end_user_formatting, newline    the two tokens that close the user turn (<|im_end|>, \\n)
    first_assistant_temp            turn_start token of the assistant header
    prompt_last                     last prompt token = last token of the header (len(cfg.header) tokens after it)
    first/last_assistant_token      first / last answer token (-1 if the answer is empty)
    last_assistant_temp             the closing token appended after the answer
answer_complete = the re-encoded answer has fewer tokens than cfg.max_new_tokens (as in the Olmo-7B run).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cases
import models


def positions(index, tok, cfg, animals):
    """index: rows of index_*.jsonl. Returns (rows, stats)."""
    before, after, user_close = cases.template_parts(tok, cfg.template_kwargs)
    prompts, spans = [], []
    for r in index:
        animal, slot, ident, w, f = r["uid"].split("|")
        frame = animals[animal]["question_frames"][slot][int(f)]
        wording = animals[animal]["identifiers"][ident]["wordings"][int(w)]
        q = frame.replace("{ID}", wording)
        id_start = len(before) + frame.index("{ID}")
        prompts.append(before + q + after)
        spans.append((id_start, id_start + len(wording), len(before) + len(q) + len(user_close)))

    enc = tok(prompts, add_special_tokens=False, return_offsets_mapping=True)
    answers = tok([r["answer"] for r in index], add_special_tokens=False).input_ids

    # Our prompt ids must be the ids the generation fed the model.
    for i in range(0, len(index), max(1, len(index) // 200)):
        q = prompts[i][len(before):len(prompts[i]) - len(after)]
        ref = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                      return_dict=True, **cfg.template_kwargs)["input_ids"]
        assert list(ref) == enc.input_ids[i], f"tokenization differs from the chat template: {index[i]['uid']}"

    rows, merged, bad_roundtrip = [], 0, 0
    for r, ids, offsets, ans, (id_start, id_end, header_start) in zip(
            index, enc.input_ids, enc.offset_mapping, answers, spans):
        ident = [i for i, (a, b) in enumerate(offsets) if a < id_end and b > id_start]
        newline = max(i for i, (a, b) in enumerate(offsets) if b <= header_start)
        n = len(ids)
        assert n - 1 == newline + 1 + len(cfg.header), f"assistant header is not {len(cfg.header)} tokens: {r['uid']}"
        close = ids[newline - 1]            # the token that closes the user turn closes this one too
        anchors = dict(start_of_ident=ident[0], end_of_ident=ident[-1],
                       end_user_formatting=newline - 1, newline=newline,
                       first_assistant_temp=newline + 1, prompt_last=n - 1,
                       first_assistant_token=n if ans else -1,
                       last_assistant_token=n + len(ans) - 1 if ans else -1,
                       last_assistant_temp=n + len(ans))
        # the last identifier token also carries frame text (e.g. "platypus" + "'s"); a leading
        # space on the first token is normal BPE and not flagged
        ident_end_merged = offsets[ident[-1]][1] > id_end
        merged += ident_end_merged
        bad_roundtrip += tok.decode(ans) != r["answer"]
        rows.append({**r, "n_prompt": n, **anchors, "ident_end_merged": ident_end_merged,
                     "answer_complete": len(ans) < cfg.max_new_tokens, "input_ids": ids + ans + [close]})

    # Template tokens must be the same in every row and have the expected text.
    seen = {}
    for name in ["end_user_formatting", "newline", "first_assistant_temp", "prompt_last", "last_assistant_temp"]:
        seen[name] = Counter(row["input_ids"][row[name]] for row in rows)
        assert len(seen[name]) == 1, f"{name} is not constant: {seen[name].most_common(3)}"
    for k, (name, want) in enumerate(cfg.header):
        got = Counter(tok.decode([row["input_ids"][row["first_assistant_temp"] + 1 + k]]) for row in rows)
        assert list(got) == [want], f"header token {name} is {dict(got)}, expected {want!r}"
    text = {n: tok.decode(list(c)) for n, c in seen.items()}
    assert text["newline"] == "\n" and text["first_assistant_temp"] == cfg.turn_start
    assert seen["last_assistant_temp"] == seen["end_user_formatting"]
    lens = [row["n_prompt"] for row in rows]
    stats = dict(rows=len(rows), prompt_min=min(lens), prompt_max=max(lens),
                 empty=sum(row["last_assistant_token"] < 0 for row in rows),
                 at_budget=sum(not row["answer_complete"] for row in rows), ident_end_merged=merged,
                 answer_decode_differs=bad_roundtrip, template_tokens=text)
    return rows, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=models.MODELS)
    ap.add_argument("acts", help="folder with index_*.jsonl")
    ap.add_argument("--out", default=".")
    ap.add_argument("--data", default="animals.json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    cfg = models.MODELS[a.model]
    index = [json.loads(line) for p in sorted(Path(a.acts).glob("index_*.jsonl")) for line in p.open()]
    rows, stats = positions(index, AutoTokenizer.from_pretrained(cfg.id), cfg, cases.load_animals(a.data))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "positions.jsonl").open("w") as fh:
        fh.writelines(json.dumps(row) + "\n" for row in rows)
    print(json.dumps(stats, indent=1, ensure_ascii=False), f"\n-> {out / 'positions.jsonl'}")
