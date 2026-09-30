#!/usr/bin/env bash
# One process per GPU. All ranks share a RID so they write into the same run directory.
set -euo pipefail

export ANIMALS_DATA="${ANIMALS_DATA:-animals.json}"
export RCLONE_DEST="${RCLONE_DEST-NotaSpider:not_a_spider}"   # set to "" to skip the Drive push
test -f "$ANIMALS_DATA" || { echo "$ANIMALS_DATA missing"; exit 1; }
if [ -f identifier_selection.json ]; then
  echo "using identifier_selection.json"
else
  echo "no identifier_selection.json -- every concept will be treated as reliable"
fi

# The RID hashes the dataset AND the identifier selection, if present. Both decide which
# prompts exist and how examples[RANK::WORLD] splits them, so a run must never resume
# across a change in either.
RID="${RID:-$(date +%Y%m%d-%H%M%S)-$(python3 -c '
import hashlib, os
h = hashlib.sha1()
for p in (os.environ["ANIMALS_DATA"], "identifier_selection.json"):
    if os.path.exists(p):
        h.update(open(p, "rb").read())
print(h.hexdigest()[:6])')}"
WORLD="${WORLD:-4}"
mkdir -p logs
echo "run id: $RID  |  data: $ANIMALS_DATA  |  world: $WORLD  |  rclone: ${RCLONE_DEST:-<disabled>}"

for r in $(seq 0 $((WORLD-1))); do
  CUDA_VISIBLE_DEVICES=$r RANK=$r WORLD=$WORLD RID=$RID \
    ANIMALS_DATA="$ANIMALS_DATA" RCLONE_DEST="$RCLONE_DEST" \
    python3 extract_acts_sharded.py > "logs/rank$r.log" 2>&1 &
done
wait
echo "all ranks finished -> acts/$RID"

# Rerunning this with the same RID resumes: each rank skips the uids already in its index.