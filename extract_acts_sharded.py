# %% config
from dataclasses import dataclass, asdict
from pathlib import Path
import json, re, os, subprocess, hashlib
from datetime import datetime
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/Olmo-3-7B-Instruct"
DATA = Path("./animals.json")
SELECTION = Path("./identifier_selection.json")   # produced by make_selection.py
SKIP_UNRELIABLE = True      # False keeps every prompt and only tags it (see Example.reliable)
OUT_DIR = Path("./acts")
MAX_NEW_TOKENS = 8
SHARD_EVERY = 2000          # ~500 MiB per shard at 33 layers x 4096 dims
RCLONE_DEST = "gdrive:acts" # "" keeps everything local (6.3 GiB total)

RANK = int(os.environ.get("RANK", 0))     # one process per GPU
WORLD = int(os.environ.get("WORLD", 1))

ANIMALS = json.loads(DATA.read_text())
BY_NAME = {a["animal"]: a for a in ANIMALS}

# Identifiers that failed the identification probe are not usable as question stems:
# the model cannot tell which animal they denote, so a wrong attribute answer is
# uninterpretable. RELIABLE holds (animal, kind, identifier_index) for the ones that passed.
_sel = json.loads(SELECTION.read_text())
RELIABLE = {(animal, kind, i)
            for animal, kinds in _sel["reliable"].items()
            for kind, idxs in kinds.items() for i in idxs}
# taxonomic_class questions are built from identifier_variants_class_neutral, every other
# attribute from identifier_variants -- the selection is per identifier list, so map the slot.
kind_of = lambda slot: "class_neutral" if slot == "taxonomic_class" else "normal"

# RID must match across ranks, so the launcher sets it; the fallback is for a single process.
# The selection is part of the hash: changing it changes which prompts exist and how they are
# split across ranks, so it must not resume into an earlier run directory.
RID = os.environ.get("RID") or (
    f"{datetime.now():%Y%m%d-%H%M%S}-"
    f"{hashlib.sha1(json.dumps([MODEL, ANIMALS, _sel['reliable']], sort_keys=True).encode()).hexdigest()[:6]}")
print(f"run {RID} | rank {RANK}/{WORLD} | {len(ANIMALS)} animals "
      f"| {len(RELIABLE)} reliable identifiers | skip_unreliable={SKIP_UNRELIABLE}")


# %% load
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map=None, attn_implementation="sdpa"
).to("cuda").eval()
print(model.dtype, model.device)


# %% model health -- run this FIRST, before anything else
out = model(**tok("hello", return_tensors="pt").to(model.device), output_hidden_states=True)
norms = [h.float().norm(dim=-1).mean().item() for h in out.hidden_states]
print("finite:", torch.isfinite(out.logits).all().item())
print("  ".join(f"L{i:02d} {n:.1f}" for i, n in enumerate(norms)))
# healthy: monotonic growth, one jump at the last layer (final norm is applied there).
# zeros / nan -> loading problem, stop here.


# %% build
@dataclass
class Example:
    uid: str
    animal: str
    kind: str           # "attribute" | "jeopardy"
    slot: str           # attribute name, or jeopardy index
    ident: int          # identifier_variants index, -1 for jeopardy
    frame: int          # question_frames index, -1 for jeopardy
    question: str
    full: str           # exactly what was fed to the model, template included
    n_tokens: int
    input_ids: list[int]
    answer: str = ""    # model output, filled in the extract cell
    correct: bool = False
    reliable: bool = True   # identifier passed the identification probe


def build(animal, kind, slot, ident, frame, question, reliable=True):
    full = tok.apply_chat_template([{"role": "user", "content": question}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(full, add_special_tokens=False).input_ids  # template already inserted BOS
    return Example(f"{animal}|{kind}|{slot}|{ident}|{frame}", animal, kind, slot,
                   ident, frame, question, full, len(ids), ids, reliable=reliable)


examples, dropped = [], 0
for a in ANIMALS:
    for slot, rows in a["attribute_questions_paraphrased"].items():
        for i, row in enumerate(rows):
            if row is None:        # identifier skipped for this attribute: it leaked the answer
                continue
            reliable = (a["animal"], kind_of(slot), i) in RELIABLE
            if not reliable and SKIP_UNRELIABLE:
                dropped += sum(q is not None for q in row)
                continue
            for f, q in enumerate(row):
                examples.append(build(a["animal"], "attribute", slot, i, f, q, reliable))
    for j, q in enumerate(a["jeopardy_questions"]):
        examples.append(build(a["animal"], "jeopardy", str(j), -1, -1, q))

lens = [e.n_tokens for e in examples]
skipped = sum(len(v) for a in ANIMALS for v in a.get("paraphrase_skipped_identifiers", {}).values())
print(f"{len(examples)} prompts ({sum(e.kind=='attribute' for e in examples)} attribute, "
      f"{sum(e.kind=='jeopardy' for e in examples)} jeopardy, {skipped} identifier rows skipped)")
print(f"dropped {dropped} prompts on unreliable identifiers"
      + ("" if SKIP_UNRELIABLE else f" (kept and tagged: {sum(not e.reliable for e in examples)})"))
print(f"tokens {min(lens)}-{max(lens)} | last token {tok.decode([examples[0].input_ids[-1]])!r}")
assert len({e.uid for e in examples}) == len(examples), "uid collision"
assert len({e.question for e in examples}) == len(examples), "duplicate question text"
assert len({e.input_ids[-1] for e in examples}) == 1, "prompt_last token differs across examples"


# %% score
def normalize(text):
    return ' '.join(re.sub(r'[^\w\s]', ' ', text.lower()).split())


def hit(candidates, answer):
    """Word-boundary search: 'bat' will not match 'wombat', and 'Four.' matches 'four'."""
    n = normalize(answer)
    return any(re.search(rf'\b{re.escape(normalize(c))}\b', n) for c in candidates)


def score(e):
    a = BY_NAME[e.animal]
    if e.kind == "jeopardy":
        return hit(a["accepted_answers"], e.answer)
    spec = a["attributes"][e.slot]
    if spec["type"] == "set":
        return hit(spec["accepted"], e.answer)
    m = re.search(r'\d+(?:\.\d+)?', e.answer)  # unit is NOT checked
    return bool(m) and spec["min"] <= float(m.group()) <= spec["max"]


# %% extract, saving a shard at a time
@torch.inference_mode()
def run(e):
    ids = torch.tensor([e.input_ids], device=model.device)
    out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                         output_hidden_states=True, return_dict_in_generate=True)
    e.answer = tok.decode(out.sequences[0, e.n_tokens:], skip_special_tokens=True).strip()
    e.correct = score(e)
    # .cpu() matters: keeping these on device would add 6.3 GiB next to the weights
    return torch.stack([h[0, -1] for h in out.hidden_states[0]]).cpu()


run_dir = OUT_DIR / RID
run_dir.mkdir(parents=True, exist_ok=True)
idx_path = run_dir / f"index_{RANK}.jsonl"
prior = [json.loads(l) for l in idx_path.read_text().splitlines()] if idx_path.exists() else []
done = {r["uid"] for r in prior}
shard = len({r["shard"] for r in prior})          # survives shards that were uploaded and unlinked
todo = [e for e in examples[RANK::WORLD] if e.uid not in done]
print(f"rank {RANK}: {len(done)} already saved, {len(todo)} to do")

buf = []


def flush():
    global buf, shard
    if not buf:
        return
    path = run_dir / f"acts_{RANK}_{shard:05d}.safetensors"
    save_file({"prompt_last": torch.stack([a for a, _ in buf]).contiguous()}, str(path))
    with idx_path.open("a") as f:                 # index is appended before any upload
        for row, (_, e) in enumerate(buf):
            f.write(json.dumps({"shard": path.name, "row": row,
                                **{k: v for k, v in asdict(e).items() if k != "input_ids"}}) + "\n")
    if RCLONE_DEST:
        subprocess.run(["rclone", "copy", str(path), RCLONE_DEST], check=True)
        path.unlink()                             # only after rclone returns 0
    shard, buf = shard + 1, []


for n, e in enumerate(todo, 1):
    buf.append((run(e), e))
    if n % SHARD_EVERY == 0:
        flush()
        print(f"rank {RANK}: {n}/{len(todo)}  ok={sum(x.correct for x in todo[:n])}")
flush()

if RANK == 0:
    (run_dir / "meta.json").write_text(json.dumps(
        {"rid": RID, "model": MODEL, "data": str(DATA), "positions": ["prompt_last"],
         "max_new_tokens": MAX_NEW_TOKENS, "n_examples": len(examples),
         "world": WORLD, "shard_every": SHARD_EVERY,
         "selection": str(SELECTION), "skip_unreliable": SKIP_UNRELIABLE,
         "n_reliable_identifiers": len(RELIABLE), "n_dropped_prompts": dropped,
         "selection_criterion": _sel.get("criterion")}, indent=2))
print(f"rank {RANK} done: {sum(e.correct for e in todo)}/{len(todo)} correct")
