"""Load pretrained GPT-2 weights (via Hugging Face) into our from-scratch model."""
from __future__ import annotations

import torch

from .model import GPT, GPTConfig

_CONV1D = ("c_attn", "c_proj", "c_fc")  # HF stores these transposed (in, out)


def from_hf(hf_model, device="cpu") -> GPT:
    c = hf_model.config
    model = GPT(GPTConfig(c.vocab_size, c.n_positions, c.n_layer, c.n_head, c.n_embd))
    sd = hf_model.state_dict()
    new = {}
    for key in model.state_dict():
        t = sd["transformer." + key.replace("blocks.", "h.")]
        if key.endswith("weight") and any(s in key for s in _CONV1D):
            t = t.t()
        new[key] = t.contiguous()
    model.load_state_dict(new)
    return model.to(device).eval()


def load_gpt2(name: str = "gpt2", device="cpu") -> GPT:
    from transformers import GPT2LMHeadModel
    return from_hf(GPT2LMHeadModel.from_pretrained(name), device)


def random_gpt2(cfg: GPTConfig | None = None, device="cpu", seed: int = 0) -> GPT:
    """Randomly initialised model: correct speed/memory behaviour, meaningless text."""
    torch.manual_seed(seed)
    return GPT(cfg or GPTConfig()).to(device).eval()


class ByteTokenizer:
    """Offline fallback tokenizer (raw bytes) used with random weights."""
    eos_token_id = None

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


def get_tokenizer(name: str = "gpt2"):
    """Real GPT-2 BPE tokenizer when weights are reachable; a byte-level fallback otherwise
    (e.g. no network, or deliberately running with --model random)."""
    if name == "random":
        return ByteTokenizer()
    try:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained(name)
        if not tok.encode("test"):  # a broken/partial offline cache can load but encode to nothing
            raise RuntimeError("tokenizer loaded but is empty")
        return tok
    except Exception:
        return ByteTokenizer()
