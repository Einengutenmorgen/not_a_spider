"""run_probe.py without PCA on a fixed set of layers. Same splits, scaler, C and CSV format.

    ACTS=/media/data/hau/acts_local python3 -u nopca_run.py SHARD WORLD
"""
import sys

import run_probe

LAYERS = ["h00", "h02", "h11", "h15", "h17", "h19", "h21", "h24", "h26", "h30", "h31"]
shard, world = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)

run_probe.feature_names = lambda *a, **k: LAYERS[shard::world]
run_probe.main(features="h", pca=0, shard=shard, world=world,
               out=f"results_{shard}.csv", base="results_nopca")
