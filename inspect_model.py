"""Inspect a model before the probing pipeline is adapted to it. Reads config + tokenizer only (no weights).

    python3 inspect_model.py allenai/Olmo-3-7B-Instruct allenai/Olmo-3.1-32B-Instruct Qwen/Qwen3.8-27B Qwen/Qwen3.5-4B > inspect.txt

Paste inspect.txt back. Nothing here is assumed: every line is read from the checkpoint.
"""
import sys

from transformers import AutoConfig, AutoTokenizer

N_CASES, D_MAX = 233_100, 2**30
SENT = "\x00QUESTION\x00"
Q = "How many limbs does one of only two kinds of monotremes, alongside the echidna have? Answer in one word."


def show(label, text):
    print(f"  {label:<34}{text!r}")


def render(tok, msgs, **kw):
    return tok.apply_chat_template(msgs, tokenize=False, **kw)


def inspect(mid):
    print(f"\n==== {mid}")
    cfg = AutoConfig.from_pretrained(mid)
    t = cfg.get_text_config()
    print("config")
    for k in ["architectures", "model_type"]:
        show(k, getattr(cfg, k, None))
    for k in ["num_hidden_layers", "hidden_size", "vocab_size", "num_attention_heads", "num_key_value_heads",
              "head_dim", "layer_types", "sliding_window", "tie_word_embeddings", "max_position_embeddings",
              "rope_theta", "rms_norm_eps", "hidden_act"]:
        if hasattr(t, k):
            v = getattr(t, k)
            show(k, v if k != "layer_types" else f"{len(v)} entries, kinds={sorted(set(v))}")
    show("dtype in config", getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None))
    show("nested text_config", t is not cfg)
    L, D = t.num_hidden_layers, t.hidden_size
    print(f"  h00..h{L - 1:02d} = hidden_states[:{L}], d01..d{L - 1:02d}; "
          f"{N_CASES * L * D * 2 / D_MAX:.0f} GiB per coordinate at {N_CASES} cases (bf16)")

    tok = AutoTokenizer.from_pretrained(mid)
    print("tokenizer")
    show("class / len(tok)", (type(tok).__name__, len(tok)))
    for k in ["bos_token", "eos_token", "pad_token", "unk_token"]:
        show(k, (getattr(tok, k), getattr(tok, k + "_id")))
    show("generation eos (generation_config)", _gen_eos(mid))
    show("padding_side", tok.padding_side)
    show("has chat_template", bool(tok.chat_template))
    tpl = tok.chat_template if isinstance(tok.chat_template, str) else str(tok.chat_template)
    show("template mentions", [w for w in ["enable_thinking", "<think>", "reasoning", "tools", "system"] if w in tpl])

    print("chat template (user message only, as the pipeline builds it)")
    variants = {"default": {}}
    if "enable_thinking" in tpl:
        variants = {"enable_thinking=default": {}, "enable_thinking=False": {"enable_thinking": False},
                    "enable_thinking=True": {"enable_thinking": True}}
    for name, kw in variants.items():
        full = render(tok, [{"role": "user", "content": SENT}], add_generation_prompt=True, **kw)
        closed = render(tok, [{"role": "user", "content": SENT}], **kw)
        before, after = full.split(SENT)
        user_close = closed.split(SENT)[1]
        print(f" [{name}]")
        show("before (system + user header)", before)
        show("after  (user close + gen prompt)", after)
        show("after starts with user close", after.startswith(user_close))
        ids = tok.apply_chat_template([{"role": "user", "content": Q}], add_generation_prompt=True, return_dict=True, **kw)["input_ids"]
        ids2 = tok(before + Q + after, add_special_tokens=False).input_ids
        show("template ids == tokenised string", list(ids) == ids2)
        n_q = len(tok(Q, add_special_tokens=False).input_ids)
        show("prompt tokens / question tokens", (len(ids), n_q))
        tail = ids[len(ids) - len(tok(after, add_special_tokens=False).input_ids) - 1:]
        show("tokens from question end on", [(i, tok.decode([i])) for i in tail])
    kw = {"enable_thinking": False} if "enable_thinking" in tpl else {}
    done = render(tok, [{"role": "user", "content": Q}, {"role": "assistant", "content": "Four"}], **kw)
    show("user+assistant('Four') rendered", done[done.rindex(Q) + len(Q):])


def _gen_eos(mid):
    try:
        from transformers import GenerationConfig
        g = GenerationConfig.from_pretrained(mid)
        return g.eos_token_id, g.pad_token_id, g.do_sample
    except Exception as e:       # no generation_config.json
        return f"unavailable ({type(e).__name__})"


if __name__ == "__main__":
    for m in sys.argv[1:]:
        try:
            inspect(m)
        except Exception as e:
            print(f"!! {m}: {type(e).__name__}: {e}")
