# Design notes

## Why these four pieces

A serving engine's speed comes from three places: not recomputing work you've already
done (KV cache), doing more work per kernel launch (batching), and moving fewer bytes
per operation (quantization). MiniServe implements the simplest correct version of each,
so the benchmark numbers can be attributed to one specific idea at a time.

## KV cache

Without a cache, generating token *t* means re-running the full forward pass on tokens
`0..t`, so step cost grows with sequence length — naive decoding is quadratic in the
number of generated tokens. The cache pre-allocates `(batch, heads, max_len, head_dim)`
buffers per layer and writes each new key/value in place (`KVCache.update` in
`miniserve/model.py`), so a decode step only computes attention for the one new query
against the stored keys. `tests/test_miniserve.py::test_cached_logits_match_at_every_step`
checks the cached and uncached logits agree exactly at every position — a cache that's
fast but wrong is worse than no cache.

## Batching

Prompts in a batch rarely have the same length, so shorter ones are left-padded and a
boolean `key_mask` marks which positions are real. The attention mask combines the causal
triangle with this padding mask (`GPT._build_mask`), and position ids are computed
per-row from the mask's cumulative sum so a padded prompt still gets correct position
embeddings. Blocked entries use the dtype's minimum value rather than `-inf`, so a row
that's entirely padding doesn't produce `NaN` after softmax.

## Quantization

`Int8Linear` stores each linear layer's weights as signed int8 with one float scale per
output row (symmetric, per-channel). Forward is `(x @ Wq^T) * scale + b` — the matmul runs
on int8, so the scale multiply is cheap and the dequantized weight matrix is never
materialized. This typically shrinks the linear-layer weights to about a quarter of their
fp32 size, with a small, measured hit to output quality (see `benchmark.py`'s `quality()`
function: KL divergence and top-1 agreement between fp32 and int8 logits on a fixed
passage).

## What's deliberately left out

Continuous batching (admitting new requests into an in-flight batch) and 4-bit
quantization are natural next steps but add real complexity for a 2-week scope. The
`docs/extensions.md`-shaped work is noted in the README instead of half-built here.
