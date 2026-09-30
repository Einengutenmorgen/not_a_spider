#!/usr/bin/env bash
# v2: generate once, then one probing process per GPU in $GPUS (default 0 1 2 3). Ranks split positions, not rows.
# Rerunning resumes: prepare skips an existing table, probe skips positions with a finished CSV.
set -euo pipefail

test -f animals.json              || { echo "animals.json missing"; exit 1; }
test -f identifier_selection.json || { echo "identifier_selection.json missing -- run make_selection.py"; exit 1; }
mkdir -p logs

# RID hashes model + animals.json + identifier_selection.json, so a changed input never
# resumes into an old run. The last stdout line of prepare is the RID.
# GPUS picks the devices; ranks are assigned in that order. GPUS="2 3" shares the box
# with another job. Run prepare only once -- two at a time would generate twice.
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
RID="${RID:-$(CUDA_VISIBLE_DEVICES=${GPUS[0]} python3 stream_probe.py prepare | tee logs/prepare.log | tail -1)}"
echo "run id: $RID on GPUs ${GPUS[*]}"

WORLD=${#GPUS[@]}
for r in $(seq 0 $((WORLD-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$r]} RANK=$r WORLD=$WORLD \
    python3 stream_probe.py probe --rid "$RID" > "logs/probe_rank$r.log" 2>&1 &
done
wait
python3 stream_probe.py summary --rid "$RID" | tee logs/summary.log