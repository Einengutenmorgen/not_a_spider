"""Mock tokenizers for the tests (no Hugging Face): single characters, plus the special tokens of the real templates."""
from tokenizers import Regex, Tokenizer, models as tm, pre_tokenizers
from transformers import PreTrainedTokenizerFast

OLMO = ("{% if messages[0]['role']!='system' %}<|im_start|>system\nSYS<|im_end|>\n{% endif %}"
        "{% for m in messages %}<|im_start|>{{m['role']}}\n{{m['content']}}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
# Qwen with the thinking switch: off -> <think> \n\n </think> \n\n in the generation prompt
# (jinja's trim_blocks eats the newline after a block tag, hence the doubled \n after the if)
QWEN = ("{% for m in messages %}<|im_start|>{{m['role']}}\n{{m['content']}}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% if enable_thinking is defined and "
        "enable_thinking is false %}\n\n</think>\n\n{% endif %}{% endif %}")
WORDS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "assistant", "\n\n"]


def mock_tok(template):
    vocab = {chr(i): i - 31 for i in range(32, 0x250)} | {"\n": 0}
    for s in WORDS + ["<unk>"]:
        vocab[s] = len(vocab)
    t = Tokenizer(tm.WordLevel(vocab, unk_token="<unk>"))
    pat = "|".join(__import__("re").escape(w) for w in WORDS) + r"|.|\n"
    t.pre_tokenizer = pre_tokenizers.Split(Regex(pat), "isolated")
    return PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", eos_token="<|im_end|>",
                                   pad_token="<unk>", chat_template=template)


ANSWERS = ["", "Four", "four limbs, so the answer is eight", "x" * 40, "It has 4 legs.", "Eight", "x" * 31,
           "about 12 cm long"]


def synthetic_index(cs, n):
    """n evenly spaced cases with made-up answers (empty, short, at the budget, with numbers)."""
    sub = cs[::len(cs) // n][:n]
    return [{k: c[k] for k in ["uid", "animal", "slot", "ident", "wording", "frame", "question", "full", "n_tokens"]}
            | {"answer": ANSWERS[i % len(ANSWERS)], "correct": i % 3 == 0, "reliable": True} for i, c in enumerate(sub)]
