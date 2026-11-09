# Runbook: probing pipeline on ipe-monster

Files live in a `hau` folder: small files in `/tmp/hau`, activations and the Hugging Face cache in `/GPUdata/hau`.

```bash
cd ~/not_a_spider && git pull && pip install pandas pyarrow
export H=/tmp/hau G=/GPUdata/hau HF_HOME=/GPUdata/hau/hf
export CUDA_VISIBLE_DEVICES=<free GPU>      # nvidia-smi: needs ~70 GB free for the 27B/32B
```

## 0. CPU tests (about 3 min, no GPU, no Hugging Face)
```bash
python3 test_cases.py && python3 test_index.py && python3 test_segments.py
```

## 1. Olmo-7B: the new code against the finished run
| Step | Command | Pass when |
|---|---|---|
| positions/tokens | done: identical | |
| plan | `MODEL_KEY=olmo7 RUN_DIR=$H/olmo7_ref ACT_DIR=$G/acts_seg/olmo7 PLAN_ONLY=1 python3 run_segments.py` then `diff $H/olmo7_ref/results_segments/plan.csv results_segments/plan.csv` | no diff (160 coordinates) |
| generation | `python3 generate.py olmo7 --limit 2000 --out $H/gen_olmo7 --compare $H/olmo7_ref/positions.jsonl` | prints the share of identical answers and of equal `correct` (batched vs the single-prompt reference; no preset threshold, read it) |
| probe | `MODEL_KEY=olmo7 RUN_DIR=$H/olmo7_ref ACT_DIR=$G/acts_seg/olmo7 ONLY_COORDS=end_of_ident python3 -u run_segments.py` then `python3 regress.py preds results_segments/preds/end_of_ident.npz $H/olmo7_ref/results_segments/preds/end_of_ident.npz` | rows, t, features identical; predictions about 100% |

`$H/olmo7_ref` holds the reference `positions.jsonl`.

## 2. Per new model (olmo32, qwen27, qwen4)
`<m>` = model key. The prompt-side coordinates (everything up to the last header token) do not depend on the answer, so
steps 2a-2b work with the Olmo-7B answers as placeholders (`$H/olmo7_idx` from `regress.py rebuild`).

```bash
# 2a. layout check on the real tokenizer, all 233,900 cases
python3 index_positions.py <m> $H/olmo7_idx --out $H/dry_<m> && python3 index_tokens.py <m> --out $H/dry_<m>

# 2b. load, hidden states, throughput, disk: one coordinate, one forward pass
MODEL_KEY=<m> RUN_DIR=$H/dry_<m> ACT_DIR=$G/acts_seg/<m> ONLY_COORDS=newline python3 -u run_segments.py

# 2c. generation: 400 cases, 50 of them again one prompt at a time
python3 generate.py <m> --limit 400 --out $H/gen_<m> --check-single 50
```
Read in the logs:
- 2b: `VRAM ok`, then `... 65 hidden states` (L+1; 33 for the 4B), the model class (Qwen: wrapper or text model?),
  last-token norms rising to the last layer. Last line: `extract Ns probe Ns` for one coordinate.
- 2c: the stop-token histogram (which token ends answers: decides the closing token of the 32B), answers at the
  budget, `single vs batched` identical share, and for Qwen whether any answer contains `<think>`.
- Qwen: a warning about the fast path for linear attention (flash-linear-attention / causal-conv1d missing) means a slower
  fallback; note the 2b time.

## 3. Full run (after the checks)
```bash
python3 generate.py <m> --out $G/acts/<m>                                    # resumable, canonical order
python3 index_positions.py <m> $G/acts/<m> --out $H/<m> && python3 index_tokens.py <m> --out $H/<m>
MODEL_KEY=<m> RUN_DIR=$H/<m> ACT_DIR=$G/acts_seg/<m> nohup python3 -u run_segments.py > $H/<m>/log 2>&1 &
```
Resumable: rerun the same command. Disk per coordinate: 57 GiB (7B), 142 GiB (27B/32B), 36 GiB (4B) times
`pos_per_pass` (models.py: 8, 8, 16). The time per pass is not known until step 2b has run:
total = number of passes x (extract + probe), with 96 coordinates / `pos_per_pass` passes for Olmo.
