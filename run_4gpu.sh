#!/usr/bin/env bash
# One process per L40. All four share a RID so they write into the same run directory.
set -euo pipefail

test -f animals.json              || { echo "animals.json missing"; exit 1; }
test -f identifier_selection.json || { echo "identifier_selection.json missing -- run make_selection.py"; exit 1; }

# The RID hashes the dataset AND the identifier selection. Both decide which prompts exist
# and how examples[RANK::WORLD] splits them, so a run must never resume across a change in either.
RID="${RID:-$(date +%Y%m%d-%H%M%S)-$(python3 -c '
import hashlib
h = hashlib.sha1()
for p in ("animals.json", "identifier_selection.json"):
    h.update(open(p, "rb").read())
print(h.hexdigest()[:6])')}"
WORLD=4
mkdir -p logs
echo "run id: $RID"

for r in $(seq 0 $((WORLD-1))); do
  CUDA_VISIBLE_DEVICES=$r RANK=$r WORLD=$WORLD RID=$RID \
    python3 extract_acts_sharded.py > "logs/rank$r.log" 2>&1 &
done
wait
echo "all ranks finished -> acts/$RID"

# Rerunning this with the same RID resumes: each rank skips the uids already in its index.
