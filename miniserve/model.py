"""GPT-2 written from scratch, with a pre-allocated KV cache and padding-aware masks."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_ctx: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class KVCache:
    """Pre-allocated key/value buffers, one (B, H, max_len, D) pair per layer.

    New keys/values are written in place, so a decode step never copies the past.
    """

    def __init__(self, cfg: GPTConfig, batch_size: int, max_len: int, device, dtype=torch.float32):
        shape = (batch_size, cfg.n_head, max_len, cfg.n_embd // cfg.n_head)
        self.k = [torch.empty(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.empty(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.length = 0

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        start, end = self.length, self.length + k.size(2)
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x, mask, cache: KVCache | None, layer: int):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            k, v = cache.update(layer, k, v)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.c_proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))  # GPT-2 uses tanh GELU


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, mask, cache, layer):
        x = x + self.attn(self.ln_1(x), mask, cache, layer)
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _build_mask(T, total, key_mask, dtype, device):
        """Additive mask (B or 1, 1, T, total). Query i may see keys j <= i + (total - T)
        that are not padding. Blocked entries get the dtype's min (not -inf), so a fully
        padded row softmaxes to uniform instead of NaN."""
        allowed = torch.ones(T, total, dtype=torch.bool, device=device).tril(total - T)[None, None]
        if key_mask is not None:
            allowed = allowed & key_mask[:, None, None, :total]
        return torch.zeros(allowed.shape, dtype=dtype, device=device).masked_fill(
            ~allowed, torch.finfo(dtype).min)

    def forward(self, idx, cache: KVCache | None = None, key_mask=None, pos_ids=None,
                last_only: bool = False):
        """idx: (B, T) new tokens. With a cache, earlier tokens are already stored.
        key_mask: (B, total) bool, False for left-padding. pos_ids: (B, T) positions."""
        B, T = idx.shape
        start = cache.length if cache is not None else 0
        total = start + T
        if pos_ids is None:
            pos_ids = torch.arange(start, total, device=idx.device).expand(B, T)
        x = self.wte(idx) + self.wpe(pos_ids)
        mask = self._build_mask(T, total, key_mask, x.dtype, idx.device)
        for i, block in enumerate(self.blocks):
            x = block(x, mask, cache, i)
        if cache is not None:
            cache.length = total
        x = self.ln_f(x)
        if last_only:
            x = x[:, -1:]
        return x @ self.wte.weight.T  # output head shares weights with the embedding
