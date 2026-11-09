"""Per-model configuration: everything model-specific that the pipeline needs, and nothing else is assumed.

The values below come from inspect_model.py output (config, tokenizer, chat template) -- see the comments.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelCfg:
    id: str
    template_kwargs: dict = field(default_factory=dict)       # passed to apply_chat_template
    stop_tokens: tuple = ("<|im_end|>", "<|endoftext|>")      # generation ends on any of these
    max_new_tokens: int = 32                                  # as in the Olmo-7B run
    gen_batch: int = 64      # prompts per generate() call; buckets share one length, so nothing is padded
    fwd_batch: int = 64      # sequences per teacher-forced pass (run_segments); decides which rows are padded
                             # together, so it is part of the setup: 64 is the Olmo-7B reference value
    pos_per_pass: int = 2    # coordinates per forward pass, limited by free disk (rows x L x D x 2 bytes each);
                             # 142 GiB each for the 27B/32B, 36 GiB for the 4B
    dtype: str = "bfloat16"
    # run_segments: condition families probed (names of data.conditions before the "/") and the coordinate
    # name prefixes left out. New models: ident + strict, no answer interior (decided); Olmo-7B: everything.
    feat_chunk: int = 2      # features fitted together in one L-BFGS run (GPU ~16 GB at 2 for the 7B); the joint
                             # run shares its line search, so this is part of the setup: 2 = Olmo-7B reference
    conditions: tuple = ("ident", "strict")
    skip_coords: tuple = ("intermediate_answer",)
    turn_start: str = "<|im_start|>"   # token that opens every turn (user header, assistant header)
    # tokens of the assistant header after turn_start, up to the last prompt token: (name, expected token text).
    # index_positions/index_tokens check the text in every case and name the coordinates after it.
    header: tuple = (("assistant_role", "assistant"), ("assistant_header_newline", "\n"))


# Qwen, thinking off: <|im_start|> assistant \n <think> \n\n </think> \n\n -- the empty think block is part of
# the generation prompt; the two "\n\n" are one token each. assistant_header_newline is the \n right after the role
# (the Olmo counterpart); the last token (Path 1's prompt_last) is think_close_newline.
QWEN_HEADER = (("assistant_role", "assistant"), ("assistant_header_newline", "\n"), ("think_open", "<think>"),
               ("think_open_newline", "\n\n"), ("think_close", "</think>"), ("think_close_newline", "\n\n"))

MODELS = {
    # reference run, finished
    "olmo7": ModelCfg("allenai/Olmo-3-7B-Instruct", conditions=("ident", "wording", "frame", "slot", "strict"),
                      skip_coords=()),
    # same vocab as the 7B, other default system prompt; generation_config lists only <|endoftext|> as eos,
    # the 7B lists both -- both are stop tokens here (as for the 7B run)
    "olmo32": ModelCfg("allenai/Olmo-3.1-32B-Instruct", gen_batch=32, fwd_batch=16, pos_per_pass=8),
    # default template = thinking mode (generation prompt ends with an open <think>): enable_thinking=False
    # gives the empty think block, so the answer starts right after the header. approved (D7).
    "qwen27": ModelCfg("Qwen/Qwen3.8-27B", {"enable_thinking": False}, gen_batch=32, fwd_batch=16, pos_per_pass=8,
                       header=QWEN_HEADER),
    "qwen4": ModelCfg("Qwen/Qwen3.5-4B", {"enable_thinking": False}, gen_batch=64, fwd_batch=32, pos_per_pass=16,
                      header=QWEN_HEADER),
}


def check_vram(cfg, headroom_gib=8):
    """Stop before loading if the free VRAM on the current device cannot hold the weights plus headroom."""
    import torch
    from huggingface_hub import HfApi
    try:
        info = HfApi().model_info(cfg.id, files_metadata=True)
        need = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors")) / 2**30
    except Exception as e:
        print(f"weights size unavailable ({type(e).__name__}): VRAM check skipped", flush=True)
        return
    free = torch.cuda.mem_get_info()[0] / 2**30
    if free < need + headroom_gib:
        raise SystemExit(f"{cfg.id}: weights {need:.0f} GiB + {headroom_gib} GiB headroom > {free:.0f} GiB free "
                         f"on {torch.cuda.get_device_name()} -- pick another GPU (CUDA_VISIBLE_DEVICES) or wait")
    print(f"VRAM ok: {free:.0f} GiB free, weights {need:.0f} GiB", flush=True)
