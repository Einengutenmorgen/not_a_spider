"""Check that the target model resolves each identifier's wordings to its animal.

Every wording of every identifier is tested under both RESOLVE_FRAMES, independently.
An identifier is accepted if >= --min-rate of its wordings resolve under frame 0
AND >= --min-rate of its wordings resolve under frame 1 (checked separately, not jointly).

python check_jeopardy.py v3 [--animals Walrus Kangaroo]
"""
import argparse, json, re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


RESOLVE_FRAMES = [
    "Name the animal described here: {ID}. Answer with its common name only.",
    "{ID} — which animal is this? Answer with its common name only.",
]

DEFAULT_MODEL = "allenai/Olmo-3-7B-Instruct"


def hit(answer, names):
    """Word-boundary match, case-insensitive, tolerates plural -s/-es."""
    return any(re.search(rf"\b{re.escape(n)}(e?s)?\b", answer, re.I) for n in names)


def load_model(model_name=DEFAULT_MODEL):
    """Load once; the refinement loop reuses this across iterations."""
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto").eval()
    return model, tok


def generate(model, tok, prompts, batch_size, max_new_tokens):
    out = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating", unit="batch"):
        chats = [tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,)
                    for p in prompts[i:i + batch_size]]

        enc = tok(
            chats,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False).to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id)

        out += tok.batch_decode(
            gen[:, enc["input_ids"].shape[1]:],
            skip_special_tokens=True)

    return [o.strip() for o in out]


def build_jobs(data, animals):
    """(animal, ident, frame_index, wording_index, prompt) for every wording x frame."""
    jobs = []
    for name in animals:
        for ident, v in data[name]["identifiers"].items():
            for j, w in enumerate(v["wordings"]):
                for f, frame in enumerate(RESOLVE_FRAMES):
                    p = frame.replace("{ID}", w)
                    jobs.append((name, ident, f, j, p[0].upper() + p[1:]))
    return jobs


def summarize(records, data, animals, min_rate):
    """Per-identifier: one pass rate per frame, computed independently."""
    summary = {}

    for name in animals:
        summary[name] = {}

        for ident in data[name]["identifiers"]:
            rs = [r for r in records if r["animal"] == name and r["ident"] == ident]

            wording_ids = sorted({r["wording"] for r in rs})
            by_wording_frame = {(r["wording"], r["frame"]): r["ok"] for r in rs}

            s = {}
            for f in range(len(RESOLVE_FRAMES)):
                passes = [by_wording_frame[(j, f)] for j in wording_ids]
                s[f"frame{f}_rate"] = sum(passes) / len(passes) if passes else 0.0
                s[f"frame{f}_failed"] = [j for j, ok in zip(wording_ids, passes) if not ok]

            s["status"] = (
                "accept"
                if all(s[f"frame{f}_rate"] >= min_rate for f in range(len(RESOLVE_FRAMES)))
                else "reject"
            )

            summary[name][ident] = s

    return summary


def run_check(data, animals, model, tok, min_rate=0.3, batch_size=32,
              max_new_tokens=32, verbose=True):
    """Generate + score + summarize. Returns (summary, records)."""
    jobs = build_jobs(data, animals)
    answers = generate(model, tok, [j[4] for j in jobs], batch_size, max_new_tokens)

    records = [dict(animal=n, ident=i, frame=f, wording=w, prompt=p, answer=ans,
                    ok=hit(ans, data[n]["alternative_naming"]))
               for (n, i, f, w, p), ans in zip(jobs, answers)]

    summary = summarize(records, data, animals, min_rate)

    if verbose:
        print()
        for name, idents in summary.items():
            for ident, s in idents.items():
                print(
                    f"{name:12} {ident:9} {s['status']:7} "
                    f"frame0={s['frame0_rate']:.1%}  frame1={s['frame1_rate']:.1%}"
                )

    return summary, records


def render_md(summary, model_name, data_path, min_rate):
    total_idents = sum(len(v) for v in summary.values())
    accepted = sum(1 for animal in summary.values()
                   for s in animal.values() if s["status"] == "accept")
    rejected = total_idents - accepted

    md = []

    md.append("# Wording Resolution Check Summary\n")
    md.append(f"**Model:** `{model_name}`  ")
    md.append(f"**Input:** `{data_path}`  ")
    md.append(f"**Minimum rate per frame:** `{min_rate:.1%}`\n")

    md.append("## Overall\n")
    md.append("| Metric | Result |")
    md.append("|---|---:|")
    md.append(f"| Identifiers tested | {total_idents} |")
    md.append(f"| Accepted | {accepted} |")
    md.append(f"| Rejected | {rejected} |")
    md.append(
        f"| Acceptance threshold | ≥ {min_rate:.1%} on frame 0 "
        f"AND ≥ {min_rate:.1%} on frame 1 |"
    )

    md.append("\n## Per-animal results\n")
    md.append("| Animal | Identifiers | Accepted | Rejected |")
    md.append("|---|---:|---:|---:|")

    for name, idents in summary.items():
        n = len(idents)
        acc = sum(s["status"] == "accept" for s in idents.values())
        md.append(f"| {name} | {n} | {acc} | {n - acc} |")

    md.append("\n## Identifier results\n")
    md.append(
        "| Animal | Identifier | Status | Frame0 rate | Frame0 failed | "
        "Frame1 rate | Frame1 failed |"
    )
    md.append("|---|---|---|---:|---|---:|---|")

    for name, idents in summary.items():
        for ident, s in idents.items():
            status = "✅ ACCEPT" if s["status"] == "accept" else "❌ REJECT"
            f0 = ", ".join(map(str, s["frame0_failed"])) if s["frame0_failed"] else "—"
            f1 = ", ".join(map(str, s["frame1_failed"])) if s["frame1_failed"] else "—"

            md.append(
                f"| {name} | `{ident}` | {status} | "
                f"{s['frame0_rate']:.1%} | {f0} | {s['frame1_rate']:.1%} | {f1} |"
            )

    if rejected:
        md.append("\n## Rejected identifiers\n")

        for name, idents in summary.items():
            failed = [i for i, s in idents.items() if s["status"] == "reject"]

            if failed:
                md.append(f"### {name}\n")
                for ident in failed:
                    s = idents[ident]
                    reason = [
                        f"frame{f} rate {s[f'frame{f}_rate']:.1%} < {min_rate:.1%}"
                        for f in range(len(RESOLVE_FRAMES))
                        if s[f"frame{f}_rate"] < min_rate
                    ]
                    md.append(f"- `{ident}` — {'; '.join(reason)}")

    return "\n".join(md)


def write_reports(out_path, summary, records, model_name, data_path, min_rate):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "min_rate": min_rate,
                "summary": summary,
                "records": records,
            },
            f, indent=2, ensure_ascii=False,
        )

    summary_out = os.path.splitext(out_path)[0] + ".md"
    with open(summary_out, "w", encoding="utf-8") as f:
        f.write(render_md(summary, model_name, data_path, min_rate))

    return out_path, summary_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version", help="Data version, e.g. v3 -> data/v3/animals.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--animals", nargs="*")
    ap.add_argument("--min-rate", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--gpu", default="1", help="CUDA_VISIBLE_DEVICES")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    run_dir = os.path.join("data", args.version)
    data_path = os.path.join(run_dir, "animals.json")
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "jeopardy_check.json")

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    animals = args.animals or list(data)

    print("\n=== TEST INPUT ===")
    print("File:", data_path)
    print("Number of animals:", len(animals))
    for name in animals:
        n_wordings = sum(len(v["wordings"]) for v in data[name]["identifiers"].values())
        print(f"{name:12} identifiers={len(data[name]['identifiers']):3} "
              f"wordings={n_wordings:4}")
    print("TOTAL:", sum(len(data[n]["identifiers"]) for n in animals), "identifiers")
    print("==================\n")

    model, tok = load_model(args.model)
    summary, records = run_check(data, animals, model, tok, args.min_rate,
                                 args.batch_size, args.max_new_tokens)

    j, m = write_reports(out_path, summary, records, args.model, data_path, args.min_rate)
    print(f"\nwrote {j}")
    print(f"wrote {m}")


if __name__ == "__main__":
    main()