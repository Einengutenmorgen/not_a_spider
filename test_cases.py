"""Mock tests (no GPU, no Hugging Face): python3 test_cases.py   -- run from the repo root."""
import json
import tempfile
from pathlib import Path

import numpy as np
from mocks import OLMO, QWEN, mock_tok

import cases as C
import generate as G

animals = C.load_animals("animals.json")
tok = mock_tok(OLMO)
cs = C.build_cases(animals, tok)

# 1 canonical order: 233,900 built; the analysis drops (data.py) leave exactly the Olmo-7B uids.npy, in its order
assert len(cs) == 233_900
keep = [c["uid"] for c in cs if (c["animal"], c["ident"]) != ("Camel", "ident_9")
        and (c["animal"], c["ident"], c["wording"]) != ("Skunk", "ident_8", 2)]
assert keep == list(np.load("results_segments/uids.npy", allow_pickle=True))

# 2 template kwargs reach the prompt (Qwen-like thinking switch); ids end on the header
q_on, q_off = mock_tok(QWEN), mock_tok(QWEN)
a = C.build_cases({k: animals[k] for k in list(animals)[:1]}, q_on, {})
b = C.build_cases({k: animals[k] for k in list(animals)[:1]}, q_off, {"enable_thinking": False})
assert a[0]["full"].endswith("<think>\n") and b[0]["full"].endswith("</think>\n\n")
assert b[0]["n_tokens"] > a[0]["n_tokens"] and a[0]["uid"] == b[0]["uid"]

# 3 scoring, lenient as in the Olmo-7B run
c = next(c for c in cs if c["slot"] == "limbs" or "limb" in c["slot"])
spec = animals[c["animal"]]["attributes"][c["slot"]]
print("slot", c["slot"], spec)
ok = {"set": spec.get("accepted", [""])[0]}
c["answer"] = f"{ok['set']}" if spec["type"] == "set" else str(int(spec["min"]))
assert C.score(animals, c)
c["answer"] = "I do not know"
assert not C.score(animals, c)

# 4 bucketed generation: no padding, resume, canonical order, schema
sub = cs[::97]
calls = []


def fake(batch):
    assert len({x["n_tokens"] for x in batch}) == 1 and len(batch) <= 7
    calls.append(len(batch))
    return [(f"ans{x['n_tokens']}", "<|im_end|>" if x["n_tokens"] % 2 else "") for x in batch]


with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    part = d / "generated.jsonl"
    done = G.run(sub[:300], 7, fake, part)
    n1 = len(calls)
    done = G.run(sub, 7, fake, part)                          # resumes: the first 300 are not regenerated
    assert sum(calls[n1:]) == len(sub) - 300 and len(done) == len(sub)
    G.finish(sub, done, animals, d, __import__("models").MODELS["olmo7"], {})
    rows = [json.loads(l) for l in (d / "index_0.jsonl").read_text().splitlines()]
    assert [r["uid"] for r in rows] == [x["uid"] for x in sub] and list(rows[0]) == G.FIELDS
print("all mock tests passed")
