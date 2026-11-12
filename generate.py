"""Greedy generation for every case, batched without padding: prompts are bucketed by exact token length.

    python3 generate.py olmo32 --out acts/<RID>                 # resumable; writes index_0.jsonl + meta.json
    python3 generate.py olmo7 --limit 2000 --out /tmp/chk --compare acts/<RID_OLMO7>/index_0.jsonl
    python3 generate.py qwen4 --out acts/<RID> --check-single 200   # single-prompt rerun of 200 cases

index_0.jsonl has the schema of extract_acts_sharded.py (without shard/row), rows in canonical order.
generated.jsonl (uid, answer, stop) is the resume file; `stop` = token that ended the answer, "" = budget.
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import cases as C
import models

FIELDS = ["uid", "animal", "slot", "ident", "wording", "frame", "question", "full", "n_tokens", "answer",
          "correct", "reliable"]


def batches(cs, size):
    by = defaultdict(list)
    for c in cs:
        by[c["n_tokens"]].append(c)
    for n in sorted(by):
        for i in range(0, len(by[n]), size):
            yield by[n][i:i + size]


def run(cs, gen_batch, gen_fn, part):
    """gen_fn(batch) -> [(answer, stop)]. Appends to `part` after every batch, skips what is there."""
    done = {}
    if part.exists():
        for line in part.read_text().splitlines():
            r = json.loads(line)
            done[r["uid"]] = r
    todo = [c for c in cs if c["uid"] not in done]
    print(f"{len(done)} cases already generated, {len(todo)} to do", flush=True)
    t0 = time.time()
    with part.open("a") as fh:
        for n, b in enumerate(batches(todo, gen_batch), 1):
            assert len({c["n_tokens"] for c in b}) == 1
            for c, (ans, stop) in zip(b, gen_fn(b)):
                r = dict(uid=c["uid"], answer=ans, stop=stop)
                done[c["uid"]] = r
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            if n % 50 == 0:
                print(f"  {len(done)}/{len(cs)}  {time.time() - t0:.0f}s", flush=True)
    if todo:
        print(f"generated {len(todo)} cases in {time.time() - t0:.0f}s ({len(todo) / (time.time() - t0):.1f} cases/s)", flush=True)
    return done


def make_gen_fn(model, tok, cfg):
    import torch
    stop = [tok.convert_tokens_to_ids(t) for t in cfg.stop_tokens]
    assert [tok.convert_ids_to_tokens(s) for s in stop] == list(cfg.stop_tokens), f"unknown stop token in {cfg.stop_tokens}"

    @torch.inference_mode()
    def gen(batch):
        x = torch.tensor([c["input_ids"] for c in batch], device=model.device)
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=cfg.max_new_tokens,
                             do_sample=False, temperature=None, top_p=None, top_k=None,
                             eos_token_id=stop, pad_token_id=tok.pad_token_id)[:, x.shape[1]:].tolist()
        res = []
        for ids in out:
            end = next((i for i, t in enumerate(ids) if t in stop), None)
            res.append((tok.decode(ids, skip_special_tokens=True).strip(),
                        "" if end is None else tok.convert_ids_to_tokens(ids[end])))
        return res
    return gen


def finish(cs, done, animals, out, cfg, extra):
    for c in cs:
        c["answer"] = done[c["uid"]]["answer"]
        c["correct"] = bool(C.score(animals, c))
    with (out / "index_0.jsonl").open("w") as fh:
        fh.writelines(json.dumps({k: c[k] for k in FIELDS}) + "\n" for c in cs)
    (out / "meta.json").write_text(json.dumps(dict(
        model=cfg.id, **{k: v for k, v in asdict(cfg).items() if k != "id"}, n_examples=len(cs), bucketed=True,
        **extra), indent=2))
    print("stop tokens:", dict(Counter(done[c["uid"]]["stop"] for c in cs)))
    print(f"correct {sum(c['correct'] for c in cs)}/{len(cs)}; at the budget "
          f"{sum(done[c['uid']]['stop'] == '' for c in cs)}")


def compare(cs, ref_path, label):
    ref = {}
    for line in open(ref_path):
        r = json.loads(line)
        ref[r["uid"]] = r
    both = [c for c in cs if c["uid"] in ref]
    same = sum(c["answer"] == ref[c["uid"]]["answer"] for c in both)
    agree = sum(c["correct"] == ref[c["uid"]]["correct"] for c in both)
    print(f"{label}: {len(both)} common cases | identical answer {same / len(both):.1%} | "
          f"same `correct` {agree / len(both):.1%}")
    for c in [c for c in both if c["answer"] != ref[c["uid"]]["answer"]][:5]:
        print(f"  {c['uid']}: {c['answer']!r} vs {ref[c['uid']]['answer']!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=models.MODELS)
    ap.add_argument("--data", default="animals.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="evenly spaced subset of the canonical order")
    ap.add_argument("--compare", help="index_*.jsonl of an earlier run: report answer / `correct` agreement")
    ap.add_argument("--check-single", type=int, default=0, help="rerun N cases one prompt at a time")
    a = ap.parse_args()

    cfg, out = models.MODELS[a.model], Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.id)
    animals = C.load_animals(a.data)
    cs = C.build_cases(animals, tok, cfg.template_kwargs)
    if a.limit:
        cs = cs[::max(1, len(cs) // a.limit)][:a.limit]
    print(f"{len(cs)} cases | prompts {min(c['n_tokens'] for c in cs)}-{max(c['n_tokens'] for c in cs)} tokens | "
          f"{len({c['n_tokens'] for c in cs})} length buckets")
    import torch
    models.check_vram(cfg)
    model = AutoModelForCausalLM.from_pretrained(cfg.id, dtype=getattr(torch, cfg.dtype), device_map=None,
                                                 attn_implementation="sdpa").to("cuda").eval()
    gen_fn = make_gen_fn(model, tok, cfg)
    done = run(cs, cfg.gen_batch, gen_fn, out / "generated.jsonl")
    finish(cs, done, animals, out, cfg, dict(limit=a.limit))
    if a.compare:
        compare(cs, a.compare, "batched vs reference")
    if a.check_single:
        sub = cs[::max(1, len(cs) // a.check_single)][:a.check_single]
        t0 = time.time()
        for c in sub:
            c["batched"] = c["answer"]
            c["answer"] = gen_fn([c])[0][0]
            c["correct"] = bool(C.score(animals, c))
        same = sum(c["answer"] == c["batched"] for c in sub)
        print(f"single vs batched: {len(sub)} cases | identical answer {same / len(sub):.1%} | "
              f"single-prompt speed {len(sub) / (time.time() - t0):.2f} cases/s")
