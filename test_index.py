"""Regression test for the template-agnostic indexing, no GPU and no Hugging Face (run from the repo root):

    python3 test_index.py

The ORIGINAL scripts (commit c696aba: OLD_index_positions.py, index_tokens.py) and the new ones run on the same mock
Olmo-like tokenizer and the same synthetic cases; positions.jsonl and tokens.parquet must be identical. Then the
Qwen-like template (thinking off: five header tokens) runs through the new scripts only.
"""
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cases as C
import index_positions as IP
import index_tokens as IT
import models
import regress
from mocks import OLMO, QWEN, mock_tok, synthetic_index

ORIGINAL = "c696aba"      # last commit before the adaptation: the Olmo-7B scripts as they ran


def git_show(path, dest):
    dest.write_text(subprocess.run(["git", "show", f"{ORIGINAL}:{path}"], capture_output=True, text=True, check=True).stdout)


with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    animals = C.load_animals("animals.json")
    shutil.copy("animals.json", d)

    # ---- Olmo-like: original scripts vs new scripts
    tokdir = d / "tok_olmo"
    mock_tok(OLMO).save_pretrained(tokdir)
    cfg = dataclasses.replace(models.MODELS["olmo7"], id=str(tokdir))
    tok = mock_tok(OLMO)
    index = synthetic_index(C.build_cases(animals, tok), 1500)
    (d / "acts").mkdir()
    (d / "acts" / "index_0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    git_show("OLD_index_positions.py", d / "old_positions.py")
    git_show("index_tokens.py", d / "old_tokens.py")
    env = {**os.environ, "MODEL": str(tokdir)}
    for script in ["old_positions.py acts", "old_tokens.py"]:
        r = subprocess.run([sys.executable, *script.split()], cwd=d, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]

    rows, stats = IP.positions(index, tok, cfg, animals)
    (d / "new").mkdir()
    (d / "new" / "positions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert regress.positions(d / "positions.jsonl", d / "new" / "positions.jsonl"), "positions.jsonl differs"
    meta, tstats = IT.tokens_table(rows, tok, cfg, animals)
    meta.to_parquet(d / "new" / "tokens.parquet", index=False)
    assert regress.tokens(d / "tokens.parquet", d / "new" / "tokens.parquet")
    print("olmo-like: identical to the original scripts |", stats["at_budget"], "answers at the budget,",
          stats["empty"], "empty")

    # rebuild: positions.jsonl -> index_0.jsonl gives back the index fields
    regress.rebuild(d / "positions.jsonl", d / "rebuilt")
    assert [json.loads(l) for l in (d / "rebuilt" / "index_0.jsonl").read_text().splitlines()] == index

    # ---- Qwen-like, thinking off: new scripts only
    qcfg = dataclasses.replace(models.MODELS["qwen4"], id="mock")
    qtok = mock_tok(QWEN)
    qindex = synthetic_index(C.build_cases(animals, qtok, qcfg.template_kwargs), 800)
    qrows, qstats = IP.positions(qindex, qtok, qcfg, animals)
    qmeta, _ = IT.tokens_table(qrows, qtok, qcfg, animals)
    r0 = qmeta[qmeta.uid == qrows[0]["uid"]]
    header = r0.loc[r0.t.between(qrows[0]["first_assistant_temp"], qrows[0]["prompt_last"]), ["token", "name"]]
    assert list(header["name"]) == ["begin_assistant_formatting", *[n for n, _ in qcfg.header]]
    assert list(header["token"]) == [qcfg.turn_start, *[t for _, t in qcfg.header]]
    for n, _ in qcfg.header:
        assert qmeta[n].sum() == len(qrows), n               # every case has every header coordinate
    assert qmeta["think_close_newline"].eq(qmeta["name"].eq("think_close_newline")).all()
    # the Olmo layout must be rejected for the Qwen config, not silently accepted
    try:
        IP.positions(index, tok, qcfg, animals)
        raise SystemExit("wrong template was accepted")
    except AssertionError:
        pass
    print("qwen-like: header", list(header["name"])[1:], "| every case has every header coordinate")
print("all index tests passed")
