# Probe design notes

Run `20260914-151738-c26147`, Olmo-3-7B-Instruct, 21,688 prompts, one activation
vector per prompt at `prompt_last`, 33 hidden states x 4096, bf16.

## 1. Input selection

**Rows.** Two exclusions from the 21,688:

- `kind == "jeopardy"` (`ident == -1`). No identifier index, so they cannot enter
  an identifier-grouped split, and the phrasing is far enough from the attribute
  frames that mixing them changes what the conditions mean.
- `slot == "taxonomic_class"`. It is the only slot built from
  `identifier_variants_class_neutral`, so its identifier index `i` refers to a
  different identifier than `i` in any other slot. Keeping it would either
  require `(animal, kind, ident)` group keys or silently mix two identifier
  pools. It is also where the pools are thinnest — Walrus has 2 reliable
  class-neutral identifiers, Armadillo 2, Nautilus 4, which cannot support a
  2-test / 2-dev holdout.

What remains is one identifier pool per animal (220 reliable identifiers over 24
animals, minimum 7 for Walrus and Toucan) x 9 slots x 10 frames, minus the
`None` rows where an identifier leaked the answer for a given attribute: about
19,500 rows.

**States.** `h[0] .. h[31]` for direct probing, `d[k] = h[k] - h[k-1]` for
k = 1..31 as residual updates.

`h[32]` is excluded from both. HF appends hidden states *before* each block and
once more after the final norm, so `h[k]` for k = 1..31 is the output of block
k-1, while `h[32]` is `norm(block 31 output)`. `h[32] - h[31]` would therefore
mix two blocks' writes with an RMSNorm rescale. Excluding it costs block 31's
output, which is only observable in normed form.

`h[32]` is still read in one place: `lens.validate()`, which needs the state the
output head actually consumes.

## 2. Formatting

Shards are row-major over examples (`[rows, 33, 4096]`), but probing wants one
state across all rows. `data.prepare()` transposes once into
`layers/h00.npy .. h32.npy`, float16, 178 MB each, 5.9 GB total. Probing a layer
off the shards directly would re-read all 5.9 GB per layer.

Row order is fixed by `build_table()`: `index_0.jsonl .. index_3.jsonl`
concatenated in filename order. Every array downstream is in that order;
`analysis_rows()` returns indices into it rather than a second copy.

float16 rather than bf16 on disk: same 2 bytes, but readable without ml_dtypes,
and every probe standardises features anyway. bf16 shards are read through a
memory-mapped view (`data.st_open`) that parses the safetensors header directly,
so no torch anywhere in the probing path.

## 3. Sampling

### The axis rule

One function builds all four conditions. Given a set of constrained axes
(`ident`, `frame`, `slot`) and their held-out values:

- **test**: held out on *every* constrained axis
- **train**: held out on *no* constrained axis
- everything in between is discarded

One axis gives a single-axis condition; three give the strict condition. Train
never contains a held-out value on any axis, so a strict test row shares no
identifier, no frame and no slot with anything the probe saw.

### Dev

Dev is a single pool defined on the identifier axis only: 2 held-out identifiers
per animal, never in train, never in test, in every condition. Model selection
therefore always happens against unseen identifiers. The cost is that dev is not
distribution-matched to the frame and slot test sets — hyperparameters chosen on
dev are chosen under identifier shift and applied under frame or slot shift.
Accepted deliberately: separate dev pools per axis would cost another slice of
training data in each condition, and the probe has few hyperparameters.

Identifiers are drawn per animal, because `ident` is only unique within an
animal. Minimum pool is 7, so 2 test / 2 dev leaves at least 3 for training
everywhere.

### Folds

`frame` holds out 2 of 10, drawn once and shared by all folds and conditions.

`slot` partitions the 9 slots into 3 groups of 3; each group is the test group in
one fold, so every slot is tested exactly once. Three folds rather than
leave-one-slot-out for two reasons: nine folds triples the fit count for the same
coverage, and a 3-slot test group makes the slot and strict conditions use the
same partition, so they are directly comparable.

Strict is the same 3 folds intersected with the identifier and frame holdouts.
Per fold its test cell is small; the three folds are disjoint in test rows and
are pooled at analysis time (weight by `n_eval` in `results.csv` — do not average
the three accuracies unweighted).

### Projected sizes

Before `None`-row deletions, from 220 identifiers / 9 slots / 10 frames:

| condition | train | dev | test |
|---|---|---|---|
| ident | 11,160 | 4,320 | 4,320 |
| frame | 8,928 | 3,456 | 3,096 |
| slot (per fold) | 7,440 | 2,880 | 5,160 |
| strict (per fold) | 5,952 | 2,304 | 288 |

Pooled strict test is ~864 rows over 24 classes — thin, and the reason the strict
number should be read with a confidence interval rather than compared to three
decimal places.

### Matched training size

The four conditions leave different amounts of training data (11,160 down to
5,952), so a raw comparison would partly measure training volume. All conditions
are subsampled to the smallest, ~5,950 rows (~248 per class).

Rows are dropped at random within class, not by group. Dropping whole groups
would equalise volume *and* group diversity, but diversity — how many distinct
identifiers, frames and slots the probe has seen — is the variable under study.
Random within-class dropping cannot leak, because the structural split has
already happened.

## 4. Controls

- **`h[0]` null.** Every prompt ends on the same token and Olmo is RoPE-only, so
  the embedding-layer row is byte-identical across all 21,688 examples. A probe
  on it collapses to the majority class — not 1/24. Asserted in `run_probe.py`
  as a row-alignment check: if `h[0]` varies across rows, the index and the
  shards are misaligned and nothing else is trustworthy.
- **Prompt length.** `n_tokens` alone as the only feature, under every condition.
  Should sit at majority-class level; if it does not, identifier length is
  class-informative and has to be residualised out.

## 5. Lens buckets

Four probabilities per row and layer: true animal's first token, the other 23
animals' first tokens, the first token the model actually generated, everything
else. The third bucket exists because these prompts ask for an attribute, so the
mass competing with the animal is the attribute answer, not noise — without it a
low `p_true` cannot be told apart from a diffuse distribution.

**Sampling.** 6,000 rows drawn uniformly (`lens.N_SAMPLE`), not all 19,500. The
buckets are per-layer means, whose standard error at n=6,000 is far below the
effect sizes of interest, while the matmul is full-vocabulary
(n x 4096 x ~100k per layer) and scales linearly in rows.

**Tokens.** Animal names are tokenised with a leading space, since that is how
they would be produced after the template; the 24 first tokens are asserted
distinct. The generated answer's first token is recovered by re-tokenising the
decoded `answer` string, because the generated ids were not stored — an
approximation that can differ from the true first id for answers beginning with
punctuation or whitespace.

Where the answer's first token *is* an animal token, the answer bucket is set to
zero rather than double-counting, so the four buckets always sum to 1.

**Normalisation.** `h[0..31]` get the final RMSNorm applied before the head;
`h[32]` does not, since it already carries it. Getting this backwards silently
produces plausible garbage.

**Validation.** `lens.validate()` checks that the unrestricted argmax at `h[32]`
reproduces the token the model actually emitted, on 200 rows. It must pass before
any earlier layer is read; failure means wrong head, wrong norm, or wrong row
alignment.

## 6. Deferred

Positional grids, span pooling and template-section averages all need a second
extraction — `positions: ["prompt_last"]` is all this run has. Correctness
prediction, geometry, jeopardy transfer and the tuned lens are out of scope here.
