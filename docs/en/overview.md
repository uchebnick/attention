# Tree Attention Overview

## Goal

Tree Attention reduces long-context decoding cost by replacing full attention over all tokens with attention over selected exact ranges and summary nodes.

Dense attention uses:

```text
scores = q @ K_all.T * scale
weights = softmax(scores)
out = weights @ V_all
```

Tree Attention uses:

```text
selected_K = concat(exact token K, summary node K)
selected_V = concat(exact token V, summary node V)
scores = q @ selected_K.T * scale
weights = softmax(scores)
out = weights @ selected_V
```

A summary node is one pseudo-token. It is not expanded over its range and is not weighted by subtree size.

## Pipeline

1. Build a semantic tree over the KV cache.
2. Store every subtree as a contiguous KV interval.
3. For each query, run range-bound tree search.
4. Return exact token intervals and summary node ids.
5. Run range attention over the compressed candidate set.

## Main tradeoff

The method is faster only if the selected candidate set is much smaller than the full context.

```text
Dense cost: O(context_tokens)
Tree decode cost: O(visited_nodes + selected_tokens + summary_tokens)
Build cost: amortized over future queries
```

## Current prototype

The current implementation has two independent layers:

- selector layer: chooses exact intervals and summary nodes;
- range attention layer: computes normal softmax attention over those candidates.

The selector can run on CPU or CUDA. The range attention layer has a CUDA kernel.
