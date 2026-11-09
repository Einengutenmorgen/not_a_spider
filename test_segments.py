"""Mock tests for run_segments.py (no GPU, no torch; run from the repo root):  python3 test_segments.py

1. rows and conditions: canonical table = the Olmo-7B uids.npy; condition filter per model
2. coordinates(): identical to the coordinate code of the ORIGINAL run_segments.py (git HEAD) for Olmo
3. coordinates() agree with the names of index_tokens.py (tokens.parquet) for the Olmo and the Qwen layout
4. coordinate filters
"""
import dataclasses
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import cases as C
import index_positions as IP
import index_tokens as IT
import models
import run_segments as RS
from mocks import OLMO, QWEN, mock_tok, synthetic_index

animals = C.load_animals("animals.json")

# 1 rows and conditions
uids, classes, y, conds = RS.table(animals, models.MODELS["olmo7"])
assert list(uids) == list(np.load("results_segments/uids.npy", allow_pickle=True))
assert [c for c, _, _ in conds] == ["ident", "wording", "frame", "slot/0", "slot/1", "slot/2",
                                    "strict/0", "strict/1", "strict/2"]
q_uids, _, _, q_conds = RS.table(animals, models.MODELS["qwen4"])
assert [c for c, _, _ in q_conds] == ["ident", "strict/0", "strict/1", "strict/2"] and list(q_uids) == list(uids)
n_match = lambda cs: min(len(i["train"]) for _, _, i in cs)
assert n_match(conds) == n_match(q_conds) == 66192          # the condition filter does not change n_match
print("table: canonical order = uids.npy; qwen conditions", [c for c, _, _ in q_conds])

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    orig = subprocess.run(["git", "show", "HEAD:run_segments.py"], capture_output=True, text=True, check=True).stdout.splitlines()
    start = next(i for i, l in enumerate(orig) if l.startswith("ANCHORS = ["))
    end = next(i for i, l in enumerate(orig) if l.startswith("COORDS = list(T)"))
    orig_coord_code = "\n".join(orig[start:end + 1])

    for name, tpl, key in [("olmo", OLMO, "olmo7"), ("qwen", QWEN, "qwen4")]:
        cfg = dataclasses.replace(models.MODELS[key], id="mock")
        tok = mock_tok(tpl)
        index = synthetic_index(C.build_cases(animals, tok, cfg.template_kwargs), 1200)
        rows, _ = IP.positions(index, tok, cfg, animals)
        meta, _ = IT.tokens_table(rows, tok, cfg, animals)
        wd = d / name
        wd.mkdir()
        (wd / "positions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        sub = np.array([r["uid"] for r in rows])
        ids, A = RS.load_positions(wd / "positions.jsonl", sub)
        T = RS.coordinates(A, cfg)

        if name == "olmo":                                   # 2 the original coordinate code, run as it was
            here = os.getcwd()
            os.chdir(wd)
            env = {"np": np, "json": json, "uids": sub}
            exec(orig_coord_code, env)
            os.chdir(here)
            assert list(env["T"]) == list(T), "coordinate names or order differ from the original"
            assert all((env["T"][c] == T[c]).all() for c in T)
            print(f"olmo: {len(T)} coordinates identical to the original run_segments.py")

        # 3 every coordinate sits on the token that index_tokens.py names the same
        lookup = meta.set_index(["uid", "t"])[["name", "name_from_end"]]
        for c, t in T.items():
            have = np.nonzero(t >= 0)[0]
            got = lookup.loc[list(zip(sub[have], t[have]))]
            assert ((got["name"] == c) | (got["name_from_end"] == c)).all(), f"{name}: {c} sits on another token"
            assert int(((meta["name"] == c) | (meta["name_from_end"] == c)).sum()) == len(have), f"{name}: {c} count"
        print(f"{name}: all {len(T)} coordinates agree with tokens.parquet names")

        # 4 filters
        cs = RS.select_coords(T, cfg)
        assert (len(cs) == len(T)) == (cfg.skip_coords == ()) and not any(c.startswith(cfg.skip_coords) for c in cs)
        assert RS.select_coords(T, cfg, ["newline", "end_of_ident"]) == [c for c in T if c in ("newline", "end_of_ident")]
        try:
            RS.select_coords(T, cfg, ["intermediate_answer_1"] if cfg.skip_coords else ["nope"])
            raise SystemExit("a skipped or unknown coordinate was accepted")
        except AssertionError:
            pass
        if name == "qwen":
            assert [c for c in cs if c in dict(cfg.header)] == [n for n, _ in cfg.header]
print("all segment tests passed")
