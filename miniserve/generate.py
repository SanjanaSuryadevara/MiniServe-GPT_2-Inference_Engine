"""Text generation: naive recompute, KV-cached, and batched (left-padded) decoding."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import GPT, KVCache


@dataclass
class GenResult:
    tokens: list[list[int]]   # generated token ids per prompt (cut at EOS)
    n_generated: int          # useful tokens across the whole batch
    seconds: float            # wall time for the whole call
    ttft: float               # time to first token (prefill), seconds

    @property
    def tokens_per_sec(self) -> float:
        return self.n_generated / self.seconds if self.seconds > 0 else 0.0


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _sample(logits, temperature, top_k, gen):
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1).cpu()
    return torch.multinomial(probs, 1, generator=gen).squeeze(1).to(logits.device)


@torch.no_grad()
def generate(model: GPT, prompts: list[list[int]], max_new_tokens: int = 64, use_cache: bool = True,
             temperature: float = 0.0, top_k: int | None = None, eos_id: int | None = None,
             seed: int = 0) -> GenResult:
    """Generate for a batch of prompts of different lengths.

    Prompts are left-padded so every sequence ends at the same column; padded keys are masked
    out and position ids are computed per row. use_cache=False recomputes the whole sequence
    at every step (the baseline the KV cache is measured against).
    """
    device = next(model.parameters()).device
    B, L = len(prompts), max(len(p) for p in prompts)
    if L + max_new_tokens > model.cfg.n_ctx:
        raise ValueError(f"prompt ({L}) + new tokens ({max_new_tokens}) exceeds context {model.cfg.n_ctx}")

    ids = torch.zeros(B, L, dtype=torch.long)
    key_mask = torch.zeros(B, L, dtype=torch.bool)
    for i, p in enumerate(prompts):
        ids[i, L - len(p):] = torch.tensor(p, dtype=torch.long)
        key_mask[i, L - len(p):] = True
    ids, key_mask = ids.to(device), key_mask.to(device)
    n_valid = key_mask.sum(1)
    pos = (key_mask.cumsum(1) - 1).clamp(min=0)

    gen = torch.Generator().manual_seed(seed)
    cache = KVCache(model.cfg, B, L + max_new_tokens, device) if use_cache else None
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    fill = eos_id if eos_id is not None else 0
    out, cur, cur_pos = [], ids, pos

    _sync(device)
    t0 = time.perf_counter()
    ttft = 0.0
    for step in range(max_new_tokens):
        logits = model(cur, cache=cache, key_mask=key_mask, pos_ids=cur_pos, last_only=True)[:, -1]
        nxt = _sample(logits, temperature, top_k, gen)
        nxt = torch.where(finished, torch.full_like(nxt, fill), nxt)
        out.append(nxt)
        if step == 0:
            _sync(device)
            ttft = time.perf_counter() - t0
        if eos_id is not None:
            finished |= nxt == eos_id
            if bool(finished.all()):
                break
        key_mask = torch.cat([key_mask, torch.ones(B, 1, dtype=torch.bool, device=device)], 1)
        if use_cache:
            cur, cur_pos = nxt[:, None], n_valid[:, None].clone()
            n_valid = n_valid + 1
        else:
            ids = torch.cat([ids, nxt[:, None]], 1)
            cur, cur_pos = ids, (key_mask.cumsum(1) - 1).clamp(min=0)
    _sync(device)
    seconds = time.perf_counter() - t0

    gen_ids = torch.stack(out, 1).tolist()
    tokens, total = [], 0
    for row in gen_ids:
        if eos_id is not None and eos_id in row:
            row = row[: row.index(eos_id) + 1]
        tokens.append(row)
        total += len(row)
    return GenResult(tokens, total, seconds, ttft)
