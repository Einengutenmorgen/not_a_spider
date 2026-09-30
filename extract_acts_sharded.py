# %% config
from dataclasses import dataclass, asdict
from pathlib import Path
import json, re, os, random, subprocess, hashlib
from datetime import datetime
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/Olmo-3-7B-Instruct"
DATA = Path(os.environ.get("ANIMALS_DATA", "./animals.json"))
SELECTION = Path("./identifier_selection.json")   # optional; see below
SKIP_UNRELIABLE = True       # only matters if SELECTION exists (see RELIABLE below)
OUT_DIR = Path(os.environ.get("OUT_DIR", "./acts"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 32))   # 8 truncated prose answers
                                                             # before the number was reached
SHARD_EVERY = 2000           # ~500 MiB per shard at 33 layers x 4096 dims
RCLONE = str(Path.home() / "bin" / "rclone")
RCLONE_DEST = os.environ.get("RCLONE_DEST", "NotaSpider:not_a_spider")  # "" disables the push

# Standard is the full v2 cross product: every concept, every wording, every attribute not
# excluded by leakage, every frame -- 10x10x10x10 minus leakage pairs, ~9.7k/animal. Set
# either to an int to subsample that axis (deterministic, seeded); None means "use all".
N_WORDINGS_PER_IDENT = None
N_FRAMES_PER_ATTR = None

RANK = int(os.environ.get("RANK", 0))     # one process per GPU
WORLD = int(os.environ.get("WORLD", 1))

# Preflight the remote before loading the model or generating anything: mkdir is idempotent,
# validates the credentials, and creates the destination if it's missing. Without this an
# expired token only surfaces at the first flush, thousands of generations in.
if RCLONE_DEST:
    _probe = subprocess.run([RCLONE, "mkdir", RCLONE_DEST], capture_output=True, text=True)
    if _probe.returncode != 0:
        raise SystemExit(
            f"rclone cannot reach {RCLONE_DEST!r}, so results could not be uploaded:\n"
            f"{_probe.stderr.strip()}\n"
            f"Fix the remote (`rclone config reconnect {RCLONE_DEST.split(':')[0]}:`) "
            f"or set RCLONE_DEST='' to keep shards on local disk only.")

# v2 schema: animals.json is a dict keyed by animal name (was a list with an "animal" key).
ANIMALS = json.loads(DATA.read_text())
BY_NAME = ANIMALS

# Resolvability check (handover section 4.3) hasn't been rerun on the new concepts yet, so
# SELECTION is optional. If present, expected shape is {"reliable": {animal: [ident_key, ...]}}
# -- flat concept keys, no more normal/class_neutral split (leakage now covers that).
if SELECTION.exists():
    _sel = json.loads(SELECTION.read_text())
    _reliable_raw = _sel["reliable"]
    _bad = sorted(a for a, v in _reliable_raw.items() if not isinstance(v, list))
    assert not _bad, (
        f"identifier_selection.json doesn't match the v2 format (a flat list of concept keys "
        f"per animal, e.g. [\"ident_1\", \"ident_3\"]) -- non-list entries for: {_bad[:5]}. "
        f"This looks like a leftover pre-rework selection file (old format was nested by "
        f"normal/class_neutral). Regenerate it for the new concepts, or delete/rename it to "
        f"run with every concept treated as reliable.")
    RELIABLE = {(animal, ident_key)
                for animal, idents in _reliable_raw.items() for ident_key in idents}
else:
    _sel = None
    RELIABLE = None   # None means "every concept is reliable" -- see build loop

# RID must match across ranks, so the launcher sets it; the fallback is for a single process.
# The selection is part of the hash: changing it changes which prompts exist and how they are
# split across ranks, so it must not resume into an earlier run directory.
RID = os.environ.get("RID") or (
    f"{datetime.now():%Y%m%d-%H%M%S}-"
    f"{hashlib.sha1(json.dumps([MODEL, ANIMALS, _sel['reliable'] if _sel else None], sort_keys=True).encode()).hexdigest()[:6]}")
print(f"run {RID} | rank {RANK}/{WORLD} | {len(ANIMALS)} animals "
      f"| {len(RELIABLE) if RELIABLE is not None else 'all'} reliable concepts "
      f"| skip_unreliable={SKIP_UNRELIABLE} "
      f"| wordings={N_WORDINGS_PER_IDENT or 'all'} frames={N_FRAMES_PER_ATTR or 'all'}")
if _sel is None:
    print("no identifier_selection.json -- every concept is being treated as reliable")


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
# Every example is an attribute question (no jeopardy -- that was for data creation, not
# probing). Fields kept for later analysis: animal, ident (concept), wording, frame, slot
# (attribute), reliable. Slice by any of these independently in the downstream analysis.
@dataclass
class Example:
    uid: str
    animal: str
    slot: str              # attribute name
    ident: str              # concept key, e.g. "ident_3" -- this is the axis "strict"/"ident"
                            # conditions hold out. Never split on `wording` instead of this.
    wording: int            # index into that concept's wordings -- bookkeeping only
    frame: int              # question_frames index
    question: str
    full: str               # exactly what was fed to the model, template included
    n_tokens: int
    input_ids: list[int]
    answer: str = ""        # model output, filled in the extract cell
    correct: bool = False
    reliable: bool = True    # concept passed the identification probe (or none has run yet)


def build(animal, slot, ident, wording, frame, question, reliable=True):
    full = tok.apply_chat_template([{"role": "user", "content": question}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(full, add_special_tokens=False).input_ids  # template already inserted BOS
    return Example(f"{animal}|{slot}|{ident}|{wording}|{frame}", animal, slot,
                   ident, wording, frame, question, full, len(ids), ids, reliable=reliable)


def sample_indices(n_total, n_pick, seed_key):
    """Deterministic, seeded subsample of n_pick indices out of range(n_total).
    n_pick=None (or >= n_total) uses every index -- the default."""
    if n_pick is None or n_pick >= n_total:
        return list(range(n_total))
    return sorted(random.Random(seed_key).sample(range(n_total), n_pick))


examples, dropped = [], 0
for animal, a in ANIMALS.items():
    for ident_key, ident in a["identifiers"].items():
        reliable = RELIABLE is None or (animal, ident_key) in RELIABLE
        wording_idx = sample_indices(len(ident["wordings"]), N_WORDINGS_PER_IDENT,
                                     f"{animal}:{ident_key}:wording")
        concept_examples = []
        for attr in a["attributes"]:
            if attr in a["leakage"].get(ident_key, []):
                continue   # this concept gives the attribute away
            frames = a["question_frames"][attr]
            frame_idx = sample_indices(len(frames), N_FRAMES_PER_ATTR, f"{animal}:{attr}:frame")
            for wi in wording_idx:
                wording = ident["wordings"][wi]
                for fi in frame_idx:
                    q = frames[fi].replace("{ID}", wording)
                    concept_examples.append(build(animal, attr, ident_key, wi, fi, q, reliable))
        if not reliable and SKIP_UNRELIABLE:
            dropped += len(concept_examples)
            continue
        examples.extend(concept_examples)

n_leak_pairs = sum(len(v) for a in ANIMALS.values() for v in a["leakage"].values())
assert examples, (
    f"0 prompts survived filtering ({dropped} dropped on unreliable concepts). Check that "
    f"identifier_selection.json actually covers the animals in {DATA} -- a selection file "
    f"built for other animals will silently drop everything here.")
lens = [e.n_tokens for e in examples]
print(f"{len(examples)} prompts | {n_leak_pairs} (concept, attribute) pairs skipped as leakage")
print(f"dropped {dropped} prompts on unreliable concepts"
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
    spec = BY_NAME[e.animal]["attributes"][e.slot]
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
    # hidden_states[0] is the prompt pass, before any token is generated; [-1] is its last
    # position -- the newline after the user turn, right before the assistant would speak.
    # Single position for path 1 by design; see the multi-position discussion for path 2.
    return torch.stack([h[0, -1] for h in out.hidden_states[0]]).cpu()


def push(local_path):
    """Copy a file to Drive without deleting the local copy. One subfolder per run, so a
    later run can't overwrite an earlier run's index_*.jsonl or meta.json."""
    if RCLONE_DEST:
        subprocess.run([RCLONE, "copy", str(local_path), f"{RCLONE_DEST}/{RID}"], check=True)


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
        push(path)
        path.unlink()      # only after rclone returns 0 -- the shard is safe to drop locally
        push(idx_path)      # BUG FIX: index grows across flushes and was never pushed before;
                            # kept locally too since resume (`done`, above) reads it back
    shard, buf = shard + 1, []


for n, e in enumerate(todo, 1):
    buf.append((run(e), e))
    if n % SHARD_EVERY == 0:
        flush()
        print(f"rank {RANK}: {n}/{len(todo)}  ok={sum(x.correct for x in todo[:n])}")
flush()

if RANK == 0:
    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(
        {"rid": RID, "model": MODEL, "data": str(DATA), "positions": ["prompt_last"],
         "max_new_tokens": MAX_NEW_TOKENS, "n_examples": len(examples),
         "world": WORLD, "shard_every": SHARD_EVERY,
         "selection": str(SELECTION) if SELECTION.exists() else None,
         "skip_unreliable": SKIP_UNRELIABLE,
         "n_reliable_concepts": len(RELIABLE) if RELIABLE is not None else None,
         "n_dropped_prompts": dropped,
         "selection_criterion": _sel.get("criterion") if _sel else None,
         "n_wordings_per_ident": N_WORDINGS_PER_IDENT,
         "n_frames_per_attr": N_FRAMES_PER_ATTR}, indent=2))
    push(meta_path)         # BUG FIX: meta.json was never pushed before
print(f"rank {RANK} done: {sum(e.correct for e in todo)}/{len(todo)} correct")