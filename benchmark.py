"""Benchmark MiniServe: naive vs KV cache vs batching vs int8.

    python benchmark.py                      # real GPT-2 weights (needs internet once)
    python benchmark.py --model random       # offline smoke test, random weights
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from miniserve import KVCache, generate, model_size_bytes, quantize_model
from miniserve.weights import get_tokenizer, load_gpt2, random_gpt2

MB = 1024 ** 2
PASSAGE = ("The quick brown fox jumps over the lazy dog. Machine learning models are trained on large "
           "datasets to recognise patterns, and inference engines make those models fast enough to use "
           "in real applications. A key trick is to cache the attention keys and values so that each "
           "new token only needs a small amount of extra computation.")


def bench(model, prompts, new_tokens, use_cache, repeats):
    generate(model, prompts, 4, use_cache=use_cache)  # warm-up
    runs = [generate(model, prompts, new_tokens, use_cache=use_cache) for _ in range(repeats)]
    best = statistics.median(r.tokens_per_sec for r in runs)
    return best, statistics.median(r.ttft for r in runs), runs[0]


@torch.no_grad()
def quality(fp, q, tok, device):
    """How much does int8 change the model? Compare next-token distributions on real text."""
    ids = torch.tensor([tok.encode(PASSAGE)], device=device)
    lf, lq = fp(ids)[0, :-1], q(ids)[0, :-1]
    target = ids[0, 1:]
    kl = F.kl_div(F.log_softmax(lq, -1), F.log_softmax(lf, -1), log_target=True, reduction="batchmean")
    return {
        "top1_agreement": (lf.argmax(-1) == lq.argmax(-1)).float().mean().item(),
        "mean_kl": kl.item(),
        "perplexity_fp32": math.exp(F.cross_entropy(lf, target).item()),
        "perplexity_int8": math.exp(F.cross_entropy(lq, target).item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", help="'gpt2' (pretrained) or 'random'")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/results.json")
    a = ap.parse_args()

    torch.manual_seed(0)
    fp = random_gpt2(device=a.device) if a.model == "random" else load_gpt2(a.model, a.device)
    q = quantize_model(copy.deepcopy(fp))
    tok = get_tokenizer(a.model)
    g = torch.Generator().manual_seed(0)
    make = lambda b: [torch.randint(0, 50000, (a.prompt_len,), generator=g).tolist() for _ in range(b)]

    runs, checks = [], {}
    base_tps = None
    plan = [("naive", fp, False, 1)]
    for b in a.batch_sizes:
        plan += [("kv_cache", fp, True, b), ("int8_kv_cache", q, True, b)]
    for name, model, use_cache, b in plan:
        prompts = make(b)
        tps, ttft, first = bench(model, prompts, a.new_tokens, use_cache, a.repeats)
        base_tps = base_tps or tps
        kv = KVCache(model.cfg, b, a.prompt_len + a.new_tokens, "meta").nbytes() if use_cache else 0
        runs.append({"name": name, "batch_size": b, "tokens_per_sec": tps, "ttft_ms": ttft * 1e3,
                     "speedup_vs_naive": tps / base_tps, "weights_mb": model_size_bytes(model) / MB,
                     "kv_cache_mb": kv / MB})
        print(f"{name:14s} bs={b:<2d} {tps:8.1f} tok/s  ttft {ttft*1e3:7.1f} ms  x{tps/base_tps:5.1f}")

    p1 = make(1)
    checks["kv_cache_matches_naive"] = (generate(fp, p1, 24, use_cache=False).tokens
                                        == generate(fp, p1, 24, use_cache=True).tokens)
    pb = make(4)
    checks["batched_matches_single"] = all(
        generate(fp, [p], 24).tokens[0] == r for p, r in zip(pb, generate(fp, pb, 24).tokens))
    a8 = generate(fp, p1, 24).tokens[0]
    b8 = generate(q, p1, 24).tokens[0]
    checks["int8_greedy_token_match"] = sum(x == y for x, y in zip(a8, b8)) / len(a8)

    result = {"meta": {"model": a.model, "device": a.device, "torch": torch.__version__,
                       "cpu": platform.processor() or platform.machine(), "threads": torch.get_num_threads(),
                       "prompt_len": a.prompt_len, "new_tokens": a.new_tokens,
                       "pretrained_weights": a.model != "random",
                       "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
              "runs": runs, "checks": checks, "quality": quality(fp, q, tok, a.device)}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("checks:", checks, "\nquality:", result["quality"], f"\nwrote {out}")
    try:
        charts(result, out.parent)
    except ImportError:
        print("matplotlib not installed, skipping charts")


def charts(result, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"naive": "#8b95a1", "kv_cache": "#0f766e", "int8_kv_cache": "#4338ca"}
    runs = result["runs"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    labels = [f"{r['name']}\nbs={r['batch_size']}" for r in runs]
    ax[0].bar(labels, [r["tokens_per_sec"] for r in runs], color=[colors[r["name"]] for r in runs])
    ax[0].set_ylabel("tokens / second (higher is better)")
    ax[0].set_title("Throughput")
    ax[1].bar(["fp32", "int8"], [runs[1]["weights_mb"], runs[2]["weights_mb"]], color=["#0f766e", "#4338ca"])
    ax[1].set_ylabel("weights (MB)")
    ax[1].set_title("Weight memory")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(folder / "benchmark.png", dpi=160)


if __name__ == "__main__":
    main()
