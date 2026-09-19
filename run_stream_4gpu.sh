#!/usr/bin/env bash
# v2: generate once on GPU 0, then one probing process per L40. Ranks split positions, not rows.
# Rerunning resumes: prepare skips an existing table, probe skips positions with a finished CSV.
set -euo pipefail

test -f animals.json              || { echo "animals.json missing"; exit 1; }
test -f identifier_selection.json || { echo "identifier_selection.json missing -- run make_selection.py"; exit 1; }
mkdir -p logs

# RID hashes model + animals.json + identifier_selection.json, so a changed input never
# resumes into an old run. The last stdout line of prepare is the RID.
RID="${RID:-$(CUDA_VISIBLE_DEVICES=0 python3 stream_probe.py prepare | tee logs/prepare.log | tail -1)}"
echo "run id: $RID"

WORLD=4
for r in $(seq 0 $((WORLD-1))); do
  CUDA_VISIBLE_DEVICES=$r RANK=$r WORLD=$WORLD \
    python3 stream_probe.py probe --rid "$RID" > "logs/probe_rank$r.log" 2>&1 &
done
wait
python3 stream_probe.py summary --rid "$RID" | tee logs/summary.log
