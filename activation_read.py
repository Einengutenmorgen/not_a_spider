from dataclasses import dataclass, asdict
from pathlib import Path
import json, re, hashlib
from datetime import datetime
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

##########################################################################################################################
# getestet in dev_box.ipynb
#output: acts/<RID>/{acts.safetensors, index.jsonl, meta.json}
# torch.bfloat16 cuda:0
# finite: True
# L00 7.5  L01 7.9  L02 8.2  L03 8.4  L04 8.0  L05 7.8  L06 7.5  L07 9.1  L08 8.6  L09 8.7  L10 8.4  L11 8.8  L12 8.9  L13 8.6  L14 8.2  L15 8.8  L16 9.4  L17 9.7  L18 10.3  L19 11.2  L20 13.3  L21 13.4  L22 13.8  L23 14.2  L24 14.5  L25 15.8  L26 15.9  L27 16.4  L28 16.7  L29 17.0  L30 17.4  L31 17.9  L32 93.9
# 175 prompts | tokens 51-72 | last token '\n'
# finite=True  2  1  2  2  2  2  3  4  5  5  6  7  9  9  10  12  13  14  16  18  20  22  25  28  30  33  36  39  43  46  52  58  96
# attribute  20/125 correct
# jeopardy   11/50 correct
#   Platypus|attribute|number_of_limbs           'Four' True
#   Platypus|attribute|color                     'Green' False
#   Platypus|attribute|weight                    'Platypus' False
#   Platypus|attribute|lifespan                  'Platypus' False
#   Platypus|attribute|height                    'Platypus' False
# 175 examples -> acts/20260911-194927-cb8f92
##########################################################################################################################

MODEL = "allenai/Olmo-3-7B-Instruct"
DATA = Path("./animals.json")
OUT_DIR = Path("./acts")
MAX_NEW_TOKENS = 8




POSITION_NAMES = ["prompt_last", "answer_first", "answer_last"]

##########################################################################################################################
##########################################################################################################################

ANIMALS = json.loads(DATA.read_text())
BY_NAME = {a["animal"]: a for a in ANIMALS}
RID = (f"{datetime.now():%Y%m%d-%H%M%S}-"
       f"{hashlib.sha1(json.dumps([MODEL, ANIMALS], sort_keys=True).encode()).hexdigest()[:6]}")
print("run id:", RID, "|", len(ANIMALS), "animals")



# %% load
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map=None, attn_implementation="sdpa"
).to("cuda").eval()
print(model.dtype, model.device)
 
 
# %% model health -- run this FIRST, before anything else
out = model(**tok("hello", return_tensors="pt").to(model.device), output_hidden_states=True)
norms = [h.float().norm(dim=-1).mean().item() for h in out.hidden_states]
print("finite:", torch.isfinite(out.logits).all().item())
print("  ".join(f"L{i:02d} {n:.1f}" for i, n in enumerate(norms)))
# healthy: monotonic growth, one jump at the last layer (final norm is applied there).
# zeros / nan -> loading problem, stop here.
 
 
# %% build
@dataclass
class Example:
    uid: str
    animal: str
    kind: str           # "attribute" | "jeopardy"
    slot: str           # attribute name, or jeopardy index
    question: str
    full: str           # exactly what was fed to the model, template included
    n_tokens: int
    input_ids: list[int]
    answer: str = ""    # model output, filled in the extract cell
    correct: bool = False
 
 
def build(animal, kind, slot, question):
    full = tok.apply_chat_template([{"role": "user", "content": question}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(full, add_special_tokens=False).input_ids  # template already inserted BOS
    return Example(f"{animal}|{kind}|{slot}", animal, kind, slot, question, full, len(ids), ids)
 
 
examples = ([build(a["animal"], "attribute", k, q)
             for a in ANIMALS for k, q in a["attribute_questions"].items()]
            + [build(a["animal"], "jeopardy", str(i), q)
               for a in ANIMALS for i, q in enumerate(a["jeopardy_questions"])])
lens = [e.n_tokens for e in examples]
print(f"{len(examples)} prompts | tokens {min(lens)}-{max(lens)} | "
      f"last token {tok.decode([examples[0].input_ids[-1]])!r}")
assert len({e.input_ids[-1] for e in examples}) == 1, "prompt_last token differs across examples"
 
 
# %% score
def normalize(text):
    return ' '.join(re.sub(r'[^\w\s]', ' ', text.lower()).split())
 
 
def hit(candidates, answer):
    """Word-boundary search: 'bat' will not match 'wombat', and 'Four.' matches 'four'."""
    n = normalize(answer)
    return any(re.search(rf'\b{re.escape(normalize(c))}\b', n) for c in candidates)
 
 
def score(e):
    a = BY_NAME[e.animal]
    if e.kind == "jeopardy":
        return hit(a["accepted_answers"], e.answer)
    spec = a["attributes"][e.slot]
    if spec["type"] == "set":
        return hit(spec["accepted"], e.answer)
    m = re.search(r'\d+(?:\.\d+)?', e.answer)  # unit is NOT checked
    return bool(m) and spec["min"] <= float(m.group()) <= spec["max"]
 
 
# %% extract
@torch.inference_mode()
def run(e):
    ids = torch.tensor([e.input_ids], device=model.device)
    out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                         output_hidden_states=True, return_dict_in_generate=True)
    e.answer = tok.decode(out.sequences[0, e.n_tokens:], skip_special_tokens=True).strip()
    e.correct = score(e)
    return torch.stack([h[0, -1] for h in out.hidden_states[0]])  # prefill step, last prompt token
 
 def run_all():   
    acts = torch.stack([run(e) for e in examples])  # [N, L+1, D]
    print(f"finite={torch.isfinite(acts).all().item()}  " +
        "  ".join(f"{x:.0f}" for x in acts.float().norm(dim=-1).mean(0)))
    for kind in ("attribute", "jeopardy"):
        sub = [e for e in examples if e.kind == kind]
        print(f"{kind:<10} {sum(e.correct for e in sub)}/{len(sub)} correct")
    for e in examples[:5]:
        print(f"  {e.uid:<44} {e.answer[:40]!r} {e.correct}")
    
    
    # %% save
    run_dir = OUT_DIR / RID
    run_dir.mkdir(parents=True, exist_ok=True)
    
    save_file({"prompt_last": acts.contiguous()}, str(run_dir / "acts.safetensors"))
    (run_dir / "index.jsonl").write_text("\n".join(
        json.dumps({"row": i, **{k: v for k, v in asdict(e).items() if k != "input_ids"}})
        for i, e in enumerate(examples)))
    (run_dir / "meta.json").write_text(json.dumps(
        {"rid": RID, "model": MODEL, "data": str(DATA), "positions": ["prompt_last"],
        "max_new_tokens": MAX_NEW_TOKENS, "n_examples": len(examples)}, indent=2))
    print(f"{len(examples)} examples -> {run_dir}")
    
if __name__ == "__main__":
    run_all()