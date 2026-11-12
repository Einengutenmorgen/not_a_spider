"""Does batching change the hidden states run_segments extracts? Cases cut at `newline` are run in one batch (as in
run_segments.extract, with attention mask) and each alone; relative difference of h00..h{L-1} at the last real token.

    M=qwen4 RUN_DIR=$H/dry_qwen4 python3 check_padding.py        # baseline: M=olmo7 RUN_DIR=$H/olmo7_ref
    LEN=36 ... python3 check_padding.py                          # no-padding control at another length

Two batches: 16 cases of mixed length (right-padded) and 16 cases of one length (no padding at all). If only the first
differs, padding is the cause; if both differ alike, it is batch-shape rounding.
"""
import os
from collections import Counter

import torch
from transformers import AutoModel

import models
import run_segments as RS

cfg = models.MODELS[os.environ["M"]]
S = RS.setup(cfg, os.environ["RUN_DIR"], "/tmp/unused", device="cuda")
model = AutoModel.from_pretrained(cfg.id, dtype=getattr(torch, cfg.dtype)).eval().to("cuda")
pad = getattr(model.config.get_text_config(), "pad_token_id", None) or 0


@torch.inference_mode()
def last_states(seqs):                                      # [L, cases, D] float32, h00..h{L-1} at the last token
    x = torch.full((len(seqs), max(map(len, seqs))), pad)
    mask = torch.zeros_like(x)
    for j, q in enumerate(seqs):
        x[j, :len(q)], mask[j, :len(q)] = torch.tensor(q), 1
    hs = model(input_ids=x.cuda(), attention_mask=mask.cuda(), output_hidden_states=True).hidden_states[:S.L]
    return torch.stack(hs)[:, torch.arange(len(seqs)), torch.tensor([len(q) - 1 for q in seqs])].float()


pool = [S.ids[i][:S.T["newline"][i] + 1] for i in range(0, len(S.ids), 100)]
common = int(os.environ.get("LEN", 0)) or Counter(map(len, pool)).most_common(1)[0][0]     # LEN=36: control length
print(f"{cfg.id}, {S.L} layers")
for name, seqs in [("mixed lengths (padded)", pool[::len(pool) // 16][:16]),
                   (f"all length {common} (no padding)", [q for q in pool if len(q) == common][:16])]:
    rel = (last_states(seqs) - torch.cat([last_states([q]) for q in seqs], 1)).norm(dim=-1)
    rel /= torch.cat([last_states([q]) for q in seqs], 1).norm(dim=-1).clamp_min(1e-6)            # [L, cases]
    print(f"{name}: worst {rel.max():.4f} | cases with a difference above 1e-4: {int((rel.max(0).values > 1e-4).sum())}/{len(seqs)}")
    print("   per case (length, padding, worst layer diff):",
          [(len(q), max(map(len, seqs)) - len(q), round(float(rel[:, j].max()), 4)) for j, q in enumerate(seqs)])