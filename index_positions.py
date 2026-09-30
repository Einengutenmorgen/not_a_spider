"""Anchor token positions for every prompt of the Path 1 extraction.

    python3 index_positions.py acts/<RID>     # the extraction folder; only index_*.jsonl is read
    -> positions.jsonl: one row per uid, the index fields plus input_ids and the anchors

Sequence = chat-templated prompt + the Path 1 answer (re-encoded) + the turn-closing token
(<|im_end|>), which Path 1 decoded away. Path 2 should feed exactly
these input_ids (one forward pass, no generation), so every anchor points at a token the
model actually sees.

Anchors (indices into input_ids):
    start_of_ident         first token of the identifier wording
    end_of_ident           last token of the identifier wording
    end_user_formatting    last token of the user turn (<|im_end|> for Olmo 3)
    newline                the \\n after it, before the assistant header starts
    first_assistant_temp   first token of the assistant header (<|im_start|>)
    prompt_last            last prompt token, the \n after "assistant" = Path 1's position
    first_assistant_token  first answer token (-1 if the answer is empty)
    last_assistant_token   last answer token (-1 if the answer is empty)
    last_assistant_temp    the <|im_end|> closing the assistant turn

answer_complete is False when the answer used the whole generation budget: the model never
produced <|im_end|> there, so last_assistant_temp is appended rather than generated.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

MODEL = os.environ.get("MODEL", "allenai/Olmo-3-7B-Instruct")
DATA = Path("animals.json")
OUT = Path("positions.jsonl")
MAX_NEW_TOKENS = 32     # Path 1 generation budget
TEMPLATE_ANCHORS = ["end_user_formatting", "newline", "first_assistant_temp", "prompt_last",
                    "last_assistant_temp"]

tok = AutoTokenizer.from_pretrained(MODEL)
animals = json.loads(DATA.read_text())
index = [json.loads(line) for p in sorted(Path(sys.argv[1]).glob("index_*.jsonl")) for line in p.open()]


def messages(q):
    return [{"role": "user", "content": q}]


# The template is identical for every row, so render it once around a sentinel.
SENTINEL = "\x00QUESTION\x00"
before, after = tok.apply_chat_template(messages(SENTINEL), tokenize=False,
                                        add_generation_prompt=True).split(SENTINEL)
user_close = tok.apply_chat_template(messages(SENTINEL), tokenize=False).split(SENTINEL)[1]
assert after.startswith(user_close), "generation prompt does not extend the user turn"

prompts, spans = [], []
for r in index:
    animal, slot, ident, w, f = r["uid"].split("|")
    frame = animals[animal]["question_frames"][slot][int(f)]
    wording = animals[animal]["identifiers"][ident]["wordings"][int(w)]
    q = frame.replace("{ID}", wording)
    id_start = len(before) + frame.index("{ID}")
    header_start = len(before) + len(q) + len(user_close)
    prompts.append(before + q + after)
    spans.append((id_start, id_start + len(wording), header_start))

enc = tok(prompts, add_special_tokens=False, return_offsets_mapping=True)
answers = tok([r["answer"] for r in index], add_special_tokens=False).input_ids

# Our prompt ids must be the ids the extraction fed the model.
for i in range(0, len(index), max(1, len(index) // 200)):
    q = prompts[i][len(before):len(prompts[i]) - len(after)]
    ref = tok.apply_chat_template(messages(q), add_generation_prompt=True, return_dict=True)["input_ids"]
    assert list(ref) == enc.input_ids[i], f"tokenization differs from the chat template: {index[i]['uid']}"

rows, merged, bad_roundtrip = [], 0, 0
for r, ids, offsets, ans, (id_start, id_end, header_start) in zip(
        index, enc.input_ids, enc.offset_mapping, answers, spans):
    ident = [i for i, (a, b) in enumerate(offsets) if a < id_end and b > id_start]
    newline = max(i for i, (a, b) in enumerate(offsets) if b <= header_start)
    n = len(ids)
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
                 "answer_complete": len(ans) < MAX_NEW_TOKENS, "input_ids": ids + ans + [close]})

# Template anchors must sit on the same token in every row; newline must be "\n".
for name in TEMPLATE_ANCHORS:
    seen = Counter(row["input_ids"][row[name]] for row in rows)
    assert len(seen) == 1, f"{name} is not constant: {seen.most_common(3)}"
    print(f"{name:22} -> {tok.decode(list(seen)):>15}  (token {next(iter(seen))})")
row = rows[0]
assert tok.decode([row["input_ids"][row["newline"]]]) == "\n"
assert row["input_ids"][row["last_assistant_temp"]] == row["input_ids"][row["end_user_formatting"]]

with OUT.open("w") as fh:
    for row in rows:
        fh.write(json.dumps(row) + "\n")

lens = [row["n_prompt"] for row in rows]
print(f"{len(rows)} rows | prompt tokens {min(lens)}-{max(lens)} | "
      f"empty answers {sum(row['last_assistant_token'] < 0 for row in rows)} | "
      f"answers at the budget {sum(not row['answer_complete'] for row in rows)} | "
      f"end_of_ident merged {merged} | answer decode != text {bad_roundtrip}")
row = rows[0]
print(f"\nexample {row['uid']}")
for name in ["start_of_ident", "end_of_ident", "end_user_formatting", "newline", "first_assistant_temp",
             "prompt_last", "first_assistant_token", "last_assistant_token", "last_assistant_temp"]:
    if row[name] >= 0:
        print(f"  {name:22} {row[name]:4d}  {tok.decode([row['input_ids'][row[name]]])!r}")
print(f"-> {OUT}")
