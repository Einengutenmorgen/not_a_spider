"""Four-bucket logit-lens readout at prompt_last.

Per row and layer the vocabulary is partitioned into
    p_true    the true animal's first token
    p_animals the other 23 animals' first tokens
    p_answer  the first token the model actually generated
    p_rest    everything else
so that a low p_true can be attributed either to a diffuse distribution or to
the state having already committed to the attribute answer.

    python lens.py           # -> lens.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import data

MODEL_DIR = Path.home() / ".cache/huggingface/hub"   # snapshot dir of the model
N_SAMPLE = 6000          # rows drawn for the lens; bucket means need far fewer
                         # rows than a probe does, and the matmul is full-vocab
CHUNK = 256


# --- model head -------------------------------------------------------------

def load_head(snapshot: Path):
    """lm_head.weight (or the tied embedding) and the final RMSNorm gain, read
    straight out of the model's safetensors. No torch, no full model load."""
    cfg = json.loads((snapshot / "config.json").read_text())
    idx = snapshot / "model.safetensors.index.json"
    if idx.exists():
        m = json.loads(idx.read_text())["weight_map"]
        where = lambda k: snapshot / m[k]
    else:
        where = lambda k: snapshot / "model.safetensors"
    key = "model.embed_tokens.weight" if cfg.get("tie_word_embeddings") else "lm_head.weight"
    W = np.asarray(data.st_open(where(key), key), dtype=np.float32)
    g = np.asarray(data.st_open(where("model.norm.weight"), "model.norm.weight"),
                   dtype=np.float32)
    return W, g, float(cfg.get("rms_norm_eps", 1e-6))


def first_token(tok, text, lead_space=False):
    ids = tok((" " if lead_space else "") + text, add_special_tokens=False).input_ids
    return ids[0]


# --- buckets ----------------------------------------------------------------

def rmsnorm(H, g, eps):
    return H / np.sqrt((H ** 2).mean(-1, keepdims=True) + eps) * g


def buckets(H, W, animal_tok, tok_true, tok_answer, g=None, eps=1e-6, chunk=CHUNK):
    """H: (n, d) hidden state. Pass g=None for h[32], which is already normed.
    Returns (n, 4) probabilities in the order true / other animals / answer / rest."""
    out = np.empty((len(H), 4), np.float32)
    ov = np.isin(tok_answer, animal_tok)      # answer token that is also an animal
    for i in range(0, len(H), chunk):
        s = slice(i, i + chunk)
        x = H[s] if g is None else rmsnorm(H[s], g, eps)
        z = x @ W.T
        z -= z.max(-1, keepdims=True)
        e = np.exp(z)
        tot = e.sum(-1)
        p_an = e[:, animal_tok].sum(-1) / tot
        p_true = e[np.arange(len(e)), tok_true[s]] / tot
        p_ans = e[np.arange(len(e)), tok_answer[s]] / tot
        out[s, 0] = p_true
        out[s, 1] = p_an - p_true
        out[s, 2] = np.where(ov[s], 0.0, p_ans)
        out[s, 3] = 1.0 - p_an - out[s, 2]
    return out


def validate(H32, W, tok_answer, sample=200):
    """At h[32] the unrestricted argmax must reproduce the token the model really
    emitted. If it does not, the head, the norm or the row alignment is wrong and
    every earlier layer is meaningless."""
    i = np.arange(min(sample, len(H32)))
    pred = (H32[i] @ W.T).argmax(-1)
    return float((pred == tok_answer[i]).mean())


def main(out="lens.csv"):
    from transformers import AutoTokenizer

    tab_all = data.build_table()
    rows, tab = data.analysis_rows(tab_all)
    classes, y = np.unique([r["animal"] for r in tab], return_inverse=True)

    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(tab), min(N_SAMPLE, len(tab)), replace=False))
    rows, tab, y = rows[pick], [tab[i] for i in pick], y[pick]

    snapshot = next(MODEL_DIR.glob("models--allenai--Olmo-3-7B-Instruct/snapshots/*"))
    W, g, eps = load_head(snapshot)
    tok = AutoTokenizer.from_pretrained(str(snapshot))

    animal_tok = np.array([first_token(tok, c, lead_space=True) for c in classes])
    assert len(set(animal_tok)) == len(classes), "two animals share a first token"
    tok_true = animal_tok[y]
    tok_answer = np.array([first_token(tok, r["answer"] or " ") for r in tab])

    acc = validate(data.feature(f"h{data.FINAL:02d}", rows), W, tok_answer)
    print(f"head check: argmax(h[32]) == generated token for {acc:.1%} of rows")
    assert acc > 0.9, "head/norm/alignment is wrong"

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "p_true", "p_other_animals", "p_answer", "p_rest"])
        for k in data.STATES:
            H = data.feature(f"h{k:02d}", rows)
            b = buckets(H, W, animal_tok, tok_true, tok_answer, g, eps).mean(0)
            w.writerow([k] + [f"{v:.5f}" for v in b])
            print(f"L{k:02d} true {b[0]:.4f}  other {b[1]:.4f}  "
                  f"answer {b[2]:.4f}  rest {b[3]:.4f}", flush=True)


if __name__ == "__main__":
    main()
