# Score-only Tree Attention report

## Architecture

The current architecture uses the tree only for routing and score approximation. It does **not** use tree-level `summary_v` as a replacement for real token values.

For a pruned span `[start:end)`, the selector emits one score:

```text
summary_score = dot(q, node_key) * scale
```

The value path applies that score to the real token values in the interval:

```text
denom += (end - start) * exp(summary_score)
out   += exp(summary_score) * V[start:end].sum(dim=0)
```

The compact-candidate version represents a summary span as:

```text
score = summary_score + log(end - start)
value = mean(V[start:end])
```

Then ordinary softmax over compact candidates is equivalent:

```text
softmax(summary_score + log(length)) * mean_v
= exp(summary_score) * V[start:end].sum / denom
```

## Current CUDA path

The best current implementation is the v69 compact score-pack path:

1. CUDA range-bound selector returns:
   - exact ranges: `range_starts`, `range_ends`, `range_counts`;
   - summary spans: `summary_starts`, `summary_ends`, `summary_scores`, `summary_counts`;
   - metrics: visited/pruned/exact/summary counts.
2. Prefix values are built once per KV head:

   ```text
   prefix_v[t + 1] = prefix_v[t] + V[t]
   ```

3. `pack_compact_scores` packs only physical candidates:
   - exact token candidates get direct dot scores and real `V`;
   - summary span candidates get `summary_score + log(length)` and prefix-sum mean `V`;
   - padding gets `-inf` score.
4. PyTorch softmax + batched matmul computes the compact attention output.

This avoids the v66 dense-overwrite bottleneck, where the tree path still performed full `q @ K.T`, full softmax, and full `weights @ V` over all `N` tokens.

## Kaggle/T4 decode benchmark

Nvidia Tesla T4, synthetic Gemma-like GQA decode, `query_len=8`, `heads=8`, `kv_heads=2`, `head_dim=64`, threshold `0.35`.

| seq len | dense ms | tree ms | speedup | output cosine | output MAE | selector ms | compact attention ms | mean physical scores | max physical scores |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64k | 2.240 | 2.716 | 0.825x | 0.98617 | — | 1.037 | 1.173 | 370.6 | 429 |
| 128k | 3.812 | 3.361 | 1.134x | 0.97960 | — | 1.047 | 1.811 | 564.8 | 656 |
| 256k | 6.612 | 5.138 | 1.287x | 0.98357 | — | 1.081 | 3.521 | 1163.8 | 2175 |
| 512k | 8.090 | 5.569 | 1.453x | 0.98340 | — | 0.987 | 4.050 | 2021.8 | 2203 |

The important result is not that the method wins everywhere. It does not: at 64k it is still slower than dense. The important result is that the compact tree path starts beating optimized dense decode at long context and improves as the context grows.

## Comparison with earlier paths

| path | main behavior | result |
|---|---|---|
| v61 custom range kernel | streamed selected ranges and summary spans directly | correct direction but too slow |
| v64 dense CUDA overwrite | overwrote summary spans inside full dense scores | fast and correct, but still full-`N` |
| v66 seq sweep | dense-overwrite up to 512k | 1.29x at 512k, range/value still grew with `N` |
| v67 compact candidates | materialized padded `[B,8448,D]` candidates | worse because `C` was much larger than actual candidates |
| v68 dynamic compact | used dynamic `C=max(exact+summary)` | better, but still materialized candidate keys |
| v69 compact score-pack | computes exact scores in pack kernel, no candidate-key tensor | best current result |

At 512k, v69 reduced range/value time from v66 `8.981 ms` to `4.050 ms`, a `2.22x` improvement for that component.

## Production status

This is a promising prototype, not production-ready.

Known blockers:

- Quality is currently validated mostly by synthetic output cosine, not by real model tasks.
- The Hugging Face/Gemma path can compare and replace decode attention, but has not yet passed a full quality suite.
- Tree build/update cost is not solved for streaming production decode.
- CUDA kernels are prototype-level; `pack_compact_scores` still computes exact dot products serially per candidate.
- No backward/training path exists.
- Layer policy is unknown: some layers may tolerate replacement, others may not.

## Can Gemma attention be replaced without fine-tuning?

Technically, yes. During decode, a Gemma-like attention module can be replaced by computing Q from the current hidden state, reading K/V from the existing cache, running Tree Attention, and passing the result through the original `o_proj`. This uses the original model weights and does not require fine-tuning to test.

But no fine-tuning does **not** mean no quality loss. The required next measurement is real-model degradation:

- continuation negative log-likelihood / perplexity delta;
- greedy generation token match rate;
- layer-local dense-vs-tree attention output MAE/cosine;
- long-context retrieval and needle tasks;
- quality versus speed at 128k/256k/512k.

Use:

```bash
python experiments/evaluate_gemma_tree_attention_quality.py \
  --model /path/to/local/gemma \
  --device cuda \
  --prompt "The capital of France is" \
  --continuation " Paris. It is known for the Eiffel Tower." \
  --max-new-tokens 8 \
  --range-distance-threshold 0.35
```

## Recommended next step

Freeze v69 as the speed baseline and run the replacement-quality gate before a larger rewrite or training work. If real-model NLL and retrieval quality stay close to dense, then the next engineering step is a cleaner CUDA implementation with warp-parallel exact dots, persistent prefix-value cache, and incremental tree updates.
