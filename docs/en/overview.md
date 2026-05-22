# Tree Attention Overview

## Goal

Tree Attention reduces long-context decode cost by replacing full attention over all cached tokens with attention over:

- exact token ranges selected by a tree search;
- summary spans that stand for many real KV tokens with one score.

Dense decode attention uses:

```text
scores = q @ K_all.T * scale
weights = softmax(scores)
out = weights @ V_all
```

Score-only Tree Attention uses:

```text
exact candidates:   score_i = q @ K[i] * scale, value_i = V[i]
summary candidate:  score_s = q @ node_key * scale + log(end - start)
                    value_s = mean(V[start:end])
weights = softmax(compact_scores)
out = weights @ compact_values
```

The `log(end - start)` term makes one summary candidate equivalent to assigning the same score to every real token in the span:

```text
softmax(summary_score + log(length)) * mean_v
= exp(summary_score) * V[start:end].sum / denom
```

## Pipeline

1. Build a semantic tree over the KV cache.
2. Store each subtree as a contiguous KV interval.
3. For each query, run range-bound tree search.
4. Return exact token intervals and summary spans.
5. Pack compact candidates on CUDA.
6. Run softmax and value aggregation over compact candidates.

## Current prototype

The current best implementation is the v69 compact score-pack path:

- selector layer: CUDA range-bound tree traversal;
- score layer: exact token scores plus summary-span scores;
- value layer: exact `V` plus summary span prefix sums over real `V`;
- compact attention layer: softmax over physical candidates only.

Synthetic Kaggle/T4 decode results show speedup only at long context:

| seq len | dense ms | tree ms | speedup | output cosine |
|---:|---:|---:|---:|---:|
| 64k | 2.240 | 2.716 | 0.825x | 0.98617 |
| 128k | 3.812 | 3.361 | 1.134x | 0.97960 |
| 256k | 6.612 | 5.138 | 1.287x | 0.98357 |
| 512k | 8.090 | 5.569 | 1.453x | 0.98340 |

## Main tradeoff

Tree Attention is useful only if the selected physical candidate count is much smaller than the full context and quality stays acceptable.

```text
Dense decode cost: O(context_tokens)
Tree decode cost:  O(visited_nodes + exact_tokens + summary_spans)
Build cost:        amortized over future decode steps
```

The current prototype is not production-ready. The next required step is real-model quality evaluation: perplexity/logit deltas, generation stability, retrieval, and long-context QA.
