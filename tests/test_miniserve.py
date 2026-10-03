import copy

import pytest
import torch

from miniserve import GPTConfig, KVCache, generate, quantize_model, model_size_bytes
from miniserve.weights import from_hf, random_gpt2

CFG = GPTConfig(vocab_size=300, n_ctx=128, n_layer=3, n_head=4, n_embd=64)


@pytest.fixture(scope="module")
def model():
    return random_gpt2(CFG, seed=1)


def test_logits_match_hugging_face():
    """Our forward pass must reproduce Hugging Face GPT-2 exactly (same random weights)."""
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    hf = GPT2LMHeadModel(GPT2Config(vocab_size=300, n_positions=128, n_embd=64, n_layer=3, n_head=4)).eval()
    ours = from_hf(hf)
    ids = torch.randint(0, 300, (2, 17))
    with torch.no_grad():
        assert torch.allclose(hf(ids).logits, ours(ids), atol=1e-4)


def test_kv_cache_matches_full_recompute(model):
    prompt = [[5, 9, 200, 3, 41, 7]]
    naive = generate(model, prompt, 20, use_cache=False)
    cached = generate(model, prompt, 20, use_cache=True)
    assert naive.tokens == cached.tokens


def test_cached_logits_match_at_every_step(model):
    ids = torch.randint(0, 300, (1, 12))
    with torch.no_grad():
        full = model(ids)
        cache = KVCache(CFG, 1, 12, ids.device)
        model(ids[:, :8], cache=cache)
        for t in range(8, 12):
            step = model(ids[:, t:t + 1], cache=cache)
            assert torch.allclose(step[0, 0], full[0, t], atol=1e-4)


def test_batched_equals_individual(model):
    prompts = [[1, 2, 3], [10, 20, 30, 40, 50, 60, 70], [99], [4, 4, 4, 4, 4]]
    batch = generate(model, prompts, 16, use_cache=True)
    for p, row in zip(prompts, batch.tokens):
        assert generate(model, [p], 16, use_cache=True).tokens[0] == row


def test_batched_without_cache_equals_with_cache(model):
    prompts = [[1, 2, 3], [10, 20, 30, 40, 50]]
    assert generate(model, prompts, 10, use_cache=False).tokens == generate(model, prompts, 10).tokens


def test_eos_stops_generation(model):
    kw = dict(temperature=1.0, top_k=50, seed=0)
    toks = generate(model, [[7, 8, 9]], 20, **kw).tokens[0]
    i = next(i for i in range(1, len(toks)) if toks[i] not in toks[:i])
    res = generate(model, [[7, 8, 9]], 50, eos_id=toks[i], **kw)
    assert res.tokens[0] == toks[: i + 1]


def test_sampling_is_seeded(model):
    a = generate(model, [[1, 2, 3]], 12, temperature=0.9, top_k=20, seed=3).tokens
    b = generate(model, [[1, 2, 3]], 12, temperature=0.9, top_k=20, seed=3).tokens
    assert a == b


def test_int8_is_smaller_and_close(model):
    q = quantize_model(copy.deepcopy(model))
    assert model_size_bytes(q) < 0.6 * model_size_bytes(model)
    ids = torch.randint(0, 300, (2, 24))
    with torch.no_grad():
        ref, approx = model(ids), q(ids)
    rel = (ref - approx).norm() / ref.norm()
    assert rel < 0.05


def test_context_overflow_raises(model):
    with pytest.raises(ValueError):
        generate(model, [[1] * 120], 50)
