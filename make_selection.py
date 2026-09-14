#!/usr/bin/env python3
"""Derive identifier_selection.json from an identifier_check.json run.

An identifier counts as reliable only if the model named the animal correctly
under BOTH probe frames. Everything else is listed under "rejected" with the
answers it produced, so the exclusion stays auditable.
"""
import json, sys
from pathlib import Path

check = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "identifier_check.json").read_text())
data = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else "animals.json").read_text())

probes = {}
for p in check["probes"]:
    probes.setdefault((p["animal"], p["kind"], p["idx"]), []).append(p)

sel, rejected = {}, []
for a in data:
    name = a["animal"]
    sel[name] = {}
    for kind in ("normal", "class_neutral"):
        keep = []
        for i in range(len(a["identifier_variants"])):
            ps = probes.get((name, kind, i))
            if ps is None:                      # not probed -> treat as unreliable
                rejected.append({"animal": name, "kind": kind, "idx": i, "answers": None})
                continue
            if all(p["resolved"] for p in ps):
                keep.append(i)
            else:
                rejected.append({"animal": name, "kind": kind, "idx": i,
                                 "answers": [p["answer"] for p in sorted(ps, key=lambda p: p["frame"])]})
        sel[name][kind] = keep

out = {
    "source_check": check.get("model", "unknown"),
    "criterion": "resolved under both probe frames",
    "frames": check.get("frames"),
    "n_reliable": sum(len(v) for m in sel.values() for v in m.values()),
    "n_rejected": len(rejected),
    "reliable": sel,
    "rejected": rejected,
}
Path("identifier_selection.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(f"{out['n_reliable']} reliable, {out['n_rejected']} rejected -> identifier_selection.json")
