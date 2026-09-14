# Animal attribute benchmark — final selection

Target model for all validation: `allenai/Olmo-3-7B-Instruct`, bf16, greedy.

## 1. What the set contains

24 animals. Each animal carries:

| field | count | used for |
|---|---|---|
| `identifier_variants` | 10 | stems for 9 of the 10 attribute questions |
| `identifier_variants_class_neutral` | 10 | stems for the `taxonomic_class` question only |
| `question_frames` | 10 per attribute | the paraphrase axis |
| `attribute_questions_paraphrased` | 10 attrs × 10 identifiers × 10 frames | the composed questions |
| `jeopardy_questions` | 2 | free-form identification control |

Design ceiling is 1000 attribute questions per animal. 20 rows are `null` because the
identifier in that slot states the answer to that attribute (recorded in
`paraphrase_skipped_identifiers`), leaving **23,800 composed attribute questions + 48 jeopardy
questions = 23,848**.

`attribute_questions_paraphrased` is a pure function of `question_frames` and the identifier
lists: `frame.replace("{ID}", identifier)` with the first character upper-cased. Any edit to an
identifier must be followed by recomposition, or the two fall out of sync.

## 2. Selection protocol

An identifier is only usable as a question stem if the model can tell which animal it denotes.
Otherwise a wrong attribute answer is uninterpretable — it conflates "doesn't know the animal's
weight" with "doesn't know which animal we mean".

Each of the 480 identifiers was probed under two frames:

1. `Name the animal described here: {ID}. Answer with its common name only.`
2. `{ID} — which animal is this? Answer with its common name only.`

**Criterion: an identifier is accepted only if both frames return an accepted answer name.**
Single-frame passes are rejected — they are unstable under exactly the paraphrase variation the
benchmark is built on.

Result: **421 of 480 accepted (87.7%)**, 59 rejected. Rejections are listed with the answers
they produced in `identifier_selection.json` under `rejected`.

## 3. Final counts

| | prompts |
|---|---|
| composed attribute questions | 23,800 |
| jeopardy questions | 48 |
| **total available** | **23,848** |
| dropped: built on rejected identifiers | −2,160 |
| **benchmark set** | **21,688** |

The 2,160 dropped split unevenly, and the asymmetry matters for triage:

- 20 rejected **normal** identifiers → 1,770 prompts (each feeds 9 attributes × 10 frames)
- 39 rejected **class-neutral** identifiers → 390 prompts (each feeds only `taxonomic_class` × 10 frames)

So a normal identifier is worth ~9× a class-neutral one. Walrus's 8 rejected class-neutral
identifiers cost 80 prompts; its 3 rejected normal identifiers cost 270.

### Per animal

| animal | kept | of | rejected ids (normal / neutral) |
|---|---|---|---|
| Kangaroo, Chameleon, Flamingo, Camel, Wombat | 1000 | 1000 | 0 / 0 |
| Narwhal, Axolotl, Bat | 990 | 1000 | 0 / 1 |
| Skunk, Cheetah | 990 | 990 | 0 / 0 |
| Peacock | 980 | 1000 | 0 / 2 |
| Ostrich | 980 | 980 | 0 / 0 |
| Giraffe | 970 | 970 | 0 / 0 |
| Elephant | 960 | 960 | 0 / 0 |
| Sloth | 880 | 990 | 1 / 2 |
| Platypus | 860 | 1000 | 1 / 5 |
| Nautilus | 850 | 1000 | 1 / 6 |
| Hippopotamus | 810 | 1000 | 2 / 1 |
| Seahorse | 790 | 980 | 2 / 2 |
| Giant Panda | 790 | 970 | 2 / 1 |
| Armadillo | 730 | 990 | 2 / 8 |
| Toucan | 720 | 990 | 3 / 0 |
| Porcupine | 720 | 1000 | 3 / 1 |
| Walrus | 650 | 990 | 3 / 8 |

Per-animal N ranges 650–1000 (mean 902). Ten animals are at 20/20 identifiers.
**Report N per animal, or subsample to 650, if any analysis compares accuracy across animals** —
otherwise differences in item count and item difficulty ride along with the comparison.

## 4. Pangolin: excluded

Pangolin was removed from the dataset entirely. Across six rewrite rounds its identifiers scored
**0/20 every time**, under six distinct strategies: physical description, behaviour, the
trafficking/traditional-medicine framing, Malay etymology, the taxonomic order (Pholidota), and
minimal terse phrasing. The model's wrong answers were stable and confident — Aardvark, Armadillo,
Rhinoceros, Alligator, monitor lizard — which points at a missing description→label mapping in the
model rather than underspecified identifiers.

This is a finding about `Olmo-3-7B-Instruct`, not a dataset defect, and should be reported as such.
The last Pangolin-containing version of the file is kept in the iteration history if it is needed.

## 5. Known limitations

- **Scorer artifacts.** Some probe failures were correct answers rejected on surface form:
  `Sloths`, `Toucans`, `A porcupine`, and answers truncated at the 32-token probe cap. Adding
  plurals to `accepted_answers` and stripping leading articles / taking the first sentence would
  recover items in both the probe and the main run. The main run's `hit()` already does word-
  boundary matching, so `bat` will not match `wombat`, but it does not handle plurals.
- **Units are not checked.** `score()` reads the first number and ignores the unit, so `1500` passes
  a kg range whether the model meant kg or pounds. Heights are cm for some animals and m for others.
- **Reveal-nulls cap 12 animals below 1000** regardless of identifier quality (Elephant 960,
  Giraffe 970, Giant Panda 970 …). Clearing them means rewriting identifiers that currently pass.
- **Reduced paraphrase diversity in the hard animals.** Getting Walrus, Armadillo and Nautilus to
  resolve required converging their identifiers on the one phrasing that works, so within those
  animals the 10 identifiers are more similar to each other than elsewhere. The identifier axis is
  therefore weaker as a variation source for exactly those animals.
- **Selection is model-specific.** `identifier_selection.json` is valid for Olmo-3-7B-Instruct
  only. Any other target model needs its own probe run; the selection file is the only thing that
  changes.

## 6. Validation history

| round | overall | normal | class-neutral | identifiers rewritten |
|---|---|---|---|---|
| 0 | 40% | 45% | 34% | — |
| 1 | 59% | 62% | 55% | 331 |
| 2 | 72% | 76% | 67% | 231 |
| 3 | 77% | 81% | 73% | 167 |
| 4 | 82% | 86% | 78% | 132 |
| 5 | 85% | 89% | 81% | 106 |
| 6 (final, Pangolin dropped) | **90%** | **95%** | **86%** | 70 |

Gains shrank each round and two rounds produced regressions (Walrus −15pt in round 3, Sloth −25pt
in round 2) because rewrites are unverified until the next probe run. The selection file, rather
than further rewriting, is what makes the set usable now.

---

# Skipping the rejected identifiers in the extraction run

## Approaches considered

**A. Prune `animals.json` directly** — set the rejected identifiers' rows to `null`.
Simplest, no code change. Rejected: it overloads `null`, which currently means "this identifier
reveals this attribute's answer", with a second meaning; it destroys the rejected identifiers
(they are useful as negative examples); and the dataset would silently differ per target model.

**B. Sidecar selection file, filtered at build time** — a separate
`identifier_selection.json` listing accepted indices, applied in the build cell.
`animals.json` stays canonical and model-agnostic; the exclusion is auditable and versioned;
swapping target models means swapping one small file. **Chosen.**

**C. Generate everything, tag, filter at analysis time** — keep all 23,848 prompts, add a
`reliable` flag. Costs ~10% more GPU time and ~630 MiB of activations, but preserves the failure
cases, which are themselves interesting: activations for prompts where the model misidentifies the
subject are a natural contrast set for a probing study. Kept as a switch (`SKIP_UNRELIABLE=False`)
rather than the default.

**D. Read `identifier_check.json` directly in the build cell** — no sidecar. Rejected: the check
file is a bulky run artifact (1000 probe records) and couples extraction to a particular probe run's
format; deriving a small explicit selection once is cleaner and reviewable in a diff.

## What was changed

Three files, appended to this handover:

**`make_selection.py`** (new) — derives `identifier_selection.json` from an
`identifier_check.json` run. Accepts an identifier only if all its probes resolved; records the
rejects with the answers they produced.

```
python3 make_selection.py identifier_check.json animals.json
# 421 reliable, 59 rejected -> identifier_selection.json
```

**`extract_acts_sharded.py`** — four changes:

1. Config gains `SELECTION` and `SKIP_UNRELIABLE`.
2. The selection loads into a `RELIABLE` set of `(animal, kind, identifier_index)`, with
   `kind_of(slot)` mapping `taxonomic_class` → `class_neutral` and everything else → `normal`.
   This mapping is the one easy thing to get wrong: the two identifier lists are indexed
   independently, so using the normal selection for `taxonomic_class` would silently gate the
   wrong prompts.
3. The build loop skips a whole identifier row when it is unreliable (or tags it when
   `SKIP_UNRELIABLE=False`). `Example` gains `reliable: bool`, which flows into the JSONL index
   via `asdict`, so the flag is available downstream either way.
4. `meta.json` records the selection path, the flag, the number of reliable identifiers and the
   number of dropped prompts, so a run directory documents its own filtering.

**`run_4gpu.sh`** — the RID now hashes `animals.json` **and** `identifier_selection.json`, and
fails fast if either is missing.

This is the important fix. The old RID hashed only `animals.json`, and resume works by uid
(`animal|kind|slot|ident|frame`). Changing the selection changes which prompts exist and therefore
how `examples[RANK::WORLD]` splits across ranks — so with the old launcher, rerunning under a
changed selection would resume into the same run directory and interleave two incompatible example
sets, with each rank skipping uids another rank had written. Hashing the selection makes that
impossible.

## Effect

```
SKIP_UNRELIABLE=True   21,688 prompts   2,160 dropped      (5,422 per rank at WORLD=4)
SKIP_UNRELIABLE=False  23,848 prompts   2,160 tagged
```

Both modes pass the script's existing `uid` / duplicate-question / prompt-last-token assertions
(dry-run verified). Nothing else in the pipeline changes: scoring, sharding, the rclone hand-off
and resume semantics are untouched.

If you want to drop only the `taxonomic_class` prompts built on rejected class-neutral identifiers
and keep everything else, gate on `kind_of(slot) == "class_neutral"` in the build loop — that is
390 prompts rather than 2,160.
