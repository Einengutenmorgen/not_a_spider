# Probe design v2 — on-the-fly probing at every position

Same model, rows, splits, matched training size and probe as v1 (`Probe_design.md`).
What changes is *where* the state is read, and that no activations are stored.

## 1. Why on the fly

v1 stored one position: 5.9 GB. Every token coordinate (~140 positions with Olmo's
system prompt) would be close to a terabyte. v2 keeps states only in RAM, a few positions at a time, and writes
probe results only.

```
python stream_probe.py prepare                            # runs/<RID>/table.jsonl
RANK=r WORLD=4 python stream_probe.py probe --rid <RID>   # one per GPU
python stream_probe.py summary --rid <RID>                # -> summary.csv
```

- **prepare** generates greedy answers once (batched, left padding). It stores the
  generated token ids and every row's anchor indices. The RID hashes model +
  `animals.json` + `identifier_selection.json`.
- **probe** processes `POS_PER_PASS` positions per pass: one teacher-forced forward over
  prompt + answer for all rows, states into RAM, all probes for those positions fitted
  on the GPU, one CSV per position, RAM freed. Ranks split positions, not rows.
  A position counts as done once its CSV exists (written via rename).
- **summary** pools the slot and strict folds weighted by `n_eval` and writes test
  accuracy for `h`, `d` and `d_minus_h`.

Cost per pass on the host: ~5 GB × `POS_PER_PASS`
(32 states × 19.6k rows × 4096 dims, float16).

## 2. Positions

**Every token of every row is probed.** Named positions are labels for comparison, not a
selection.

### Coordinates

Each token gets a coordinate `<segment>_<offset>`, counted from the start of its segment:

| segment | content | length |
|---|---|---|
| `sys` | `<|im_start|>system\n…<|im_end|>\n<|im_start|>user\n` | fixed, identical in all rows |
| `pre` | question text before the identifier | varies with the frame (0 if the frame starts with `{ID}`) |
| `id` | the identifier | varies |
| `post` | question text after the identifier | varies with the frame |
| `tmpl` | `<|im_end|>\n<|im_start|>assistant\n` | fixed, identical in all rows |
| `ans` | generated tokens, `<|im_end|>` included | 1–8 |

- **Segment rule.** A token belongs to the segment that holds its last character, so
  `" the"` opening an identifier is `id_00`.
- **Uniqueness.** Every token of every row maps to exactly one coordinate (asserted in
  the dry run).
- **Why segment-relative.** Offsets restart at each segment, so `id_03` is the 4th
  identifier token in every row. The identifier is not shifted around by frame length.
- **Asserted identity.** The `sys` and `tmpl` token sequences are asserted identical
  across rows.

### Coverage

- **Partial coverage.** Variable segments mean that e.g. `id_20` exists only in rows with
  long identifiers. Such a coordinate is probed on the rows that have it; the others are
  dropped before fitting.
- **Held-out values do not change.** The global splits (held-out identifiers, frames,
  slots) are intersected with those rows.
- **Matched `n_train`.** It is re-matched across the conditions that keep at least
  `MIN_TRAIN` = 240 training rows (10 per class). A condition below that, or with no test
  rows, is skipped at that position.
- **Full coverage is unchanged.** At full coverage this reproduces the global split
  exactly (strict n_train 5342 / 5880 / 5334 as in v1).
- **Reporting.** `positions.csv` lists `coverage`, `n_distinct_tokens` and the most
  common token per position. `summary.csv` carries `coverage` on every row.
- **Caution.** Low-coverage coordinates are skewed towards animals with long identifiers
  or frames. Read them with their `coverage` and `n_train` next to them.

### Markers

| marker | Olmo token | coordinate |
|---|---|---|
| `seq_begin` | first `<|im_start|>` | `sys_00` |
| `ident_last` | last identifier token (`gesture`) | moves: `id_k` |
| `end_user` | `\n` after the user's `<|im_end|>` | `tmpl_01` |
| `assistant_tmp_begin` | `<|im_start|>` of the assistant header | `tmpl_02` |
| `assistant_tmp_end` | last prompt token (`\n` after `assistant`) | last `tmpl` |
| `answer_first` | first generated token | `ans_00` |
| `answer_last` | last generated token before `<|im_end|>` | moves |
| `last` | final token of the sequence (`<|im_end|>`, or the token at the 8-token cap) | moves |

- **Fixed markers are labels.** A marker that sits on the same coordinate in every row is
  a label on that coordinate (`markers` column in `positions.csv`).
- **Moving markers are probed separately.** `ident_last`, `answer_last` and `last` are
  probed as positions of their own, in addition to the coordinates they sit on.
- **How the header is found.** `end_user` and `assistant_tmp_begin` are located from the
  template: the generation prompt starts where `add_generation_prompt=False` stops.
  `prepare` prints the decoded marker tokens of the first row. Check them on the real
  tokenizer before probing.

**v1 `prompt_last` = `assistant_tmp_end`.** `extract_acts_sharded.py` stored
`h[0, -1]` of the prefill pass, i.e. the last prompt token, which is the `\n` after
`assistant`, not `end_user`. The regression check against `results/v.01` is therefore
on `assistant_tmp_end`.

### Reading the positions

- **Answer tokens.** They are the attribute answer, which is correlated with the animal.
- **`h00` is the token-identity baseline.** `h00` at a position is the embedding of its
  input token. Compare later layers against it, not against chance.
- **Null positions.** Coordinates with one distinct token and a constant prefix (all of
  `sys`) are nulls.

## 3. Hidden states vs residual updates

- **Features.** Every position is probed on `h00..h31` and `d01..d31`, where
  `d[k] = h[k] − h[k−1]` is the write of block k−1. Both use the same splits and the
  same instrument.
- **Excluded state.** `h[32]` is excluded for the same reason as in v1: it is normed.
- **Output.** `summary.csv` puts `h`, `d` and `d_minus_h` side by side per
  position × condition × layer.
- **Precision.** `d` is computed in float32 from float16 host copies. bf16 → fp16 is
  lossless for these magnitudes.

## 4. Instrument

StandardScaler → PCA(256) → L2 multinomial logistic regression (C = 1), in torch on the
GPU, batched over `LAYER_CHUNK` features.

- **Solver.** The objective is separable across features, so one L-BFGS run solves the
  whole batch.
- **Equivalence to sklearn.**
  - Standardisation uses ddof = 0 and sklearn's zero-variance rule.
  - PCA is randomized (`svd_lowrank`), as sklearn's `auto` picks at this size.
  - On synthetic data, held-out accuracy matches the v1 sklearn pipeline within ~1 point.
- **Regression check.** `assistant_tmp_end` must reproduce `results/v.01` within that margin
  before anything else is read.

## 5. Fixed batch shape

Every forward batch has the same shape:

- **Batch size.** `FWD_BATCH` rows; the last batch is padded with copies of a real row,
  which are discarded.
- **Length.** Every sequence is right-padded to the global maximum length.
- **Row order.** Rows are shuffled with a fixed seed.

Why this matters:

- **The risk.** GPU kernels (cuBLAS algorithm choice, sdpa path) can depend on tensor
  shape, so the same row can come out with tiny numeric differences depending on which
  batch it was in. If batches were length-sorted, as is usual for speed, those
  differences would correlate with prompt length. Prompt length correlates with the
  identifier and so with the animal.
- **Where it bites.** At positions where every row has the same input (all of `sys`
  and `tmpl`), the real variance is
  zero or small. StandardScaler would blow this numeric noise up to unit variance, and a
  probe could decode the class from it.
- **The fix.** Fixed shapes remove the shape dependence. Shuffling removes any remaining
  link between batch membership and class.
- **The cost.** ~20 % extra compute from padding every sequence to the maximum length.
- **Masking.** Right padding under causal attention cannot affect real tokens, so no
  attention mask is passed. This keeps every batch on the same kernel path.
- **Checks at run time.**
  - `seq_begin` (`sys_00`) must show zero spread across rows at every state; the probe log prints
    it. Non-zero values mean the kernels are not batch-invariant, and the
    identical-input positions have to be read with that in mind.
  - Wherever all rows feed the same token, `h[0]` is asserted identical. A failure means
    rows and states are misaligned.

## 6. Differences from v1 to keep in mind

- **Generation cap.** `MAX_NEW_TOKENS` is 48, not 8. With 8, most "number of years" and
  "number and unit" answers were cut off while the model restated the subject (e.g.
  `The big-footed Australian marsupial`). In those rows `answer_last` and `last` sat on
  the cap and `correct` was wrong. v1's `correct` has the same problem.
  - Rows that still hit the cap are flagged `capped` in `table.jsonl`.
  - `prepare` warns if they exceed 1 %.
  - The cap is part of the RID.
  - `ans_k` coordinates beyond a few tokens have low coverage and fall under the
    `MIN_TRAIN` rule.
- **Answers may differ.** Answers come from batched generation and can differ from
  v1's unbatched run in a few rows.
- **No re-tokenisation.** Generated ids are stored, which removes the re-tokenisation
  approximation `lens.py` needed.
- **Row set.** Rows are the v1 analysis rows only: no jeopardy, no `taxonomic_class`,
  reliable identifiers. Row identifiers use the v1 `uid` format, so v1 predictions can
  be joined.
- **Nothing is kept.** A new probe configuration means another forward pass per
  position group.