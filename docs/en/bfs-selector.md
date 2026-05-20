# BFS Range-Bound Selector

## Purpose

The selector decides which tree regions should be attended exactly and which should be replaced by summary nodes.

It does not compute attention output. It only produces intervals and summary ids.

## Node test

For query `q` and node key `k`:

```text
similarity = dot(q, k) / max(norm(q), eps)
distance   = max(0, 1 - similarity)
lower      = max(0, distance - radius)
upper      = distance + radius
```

Decision rule:

```text
if upper <= threshold:
    accept node interval as exact tokens
elif lower > threshold:
    accept node summary as one pseudo-token
elif node is leaf:
    accept leaf interval if exact budget remains, else summary
else:
    enqueue children
```

## BFS frontier model

The CUDA selector runs a level-by-level BFS.

State:

```text
frontier_old [Q, frontier_capacity]
frontier_new [Q, frontier_capacity]
old_counts   [Q]
new_counts   [Q]
```

One warp processes one node. The warp computes the query-node score, applies the bound test, and either writes an output candidate or enqueues children.

## Atomic queues

Children are first appended to a shared-memory local frontier buffer with `atomicAdd`. When needed, the local buffer is flushed to the global frontier with another atomic counter.

This avoids Python traversal and keeps per-query search on GPU.

## Budgets

The selector enforces limits:

```text
max_exact_tokens
max_summary_tokens
max_visited_nodes
max_ranges
frontier_capacity
```

If exact budget is exhausted for a leaf, the selector can use a summary node instead.

## Output order

CUDA atomics can produce a different order than CPU DFS. The attention result is order-independent as long as the selected candidate set is equivalent, because softmax is computed over the set of candidates.
