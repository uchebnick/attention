# BFS Range-Bound Selector

## Назначение

Selector решает, какие области дерева считать exact, а какие заменить summary nodes.

Он не считает attention output. Он только возвращает intervals и summary ids.

## Node test

Для query `q` и node key `k`:

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

CUDA selector делает level-by-level BFS.

State:

```text
frontier_old [Q, frontier_capacity]
frontier_new [Q, frontier_capacity]
old_counts   [Q]
new_counts   [Q]
```

Один warp обрабатывает один node. Warp считает query-node score, применяет bound test и либо пишет output candidate, либо добавляет children в queue.

## Atomic queues

Children сначала добавляются в shared-memory local frontier buffer через `atomicAdd`. Потом local buffer flush-ится в global frontier через другой atomic counter.

Так поиск остаётся на GPU и не требует Python traversal.

## Budgets

Selector соблюдает лимиты:

```text
max_exact_tokens
max_summary_tokens
max_visited_nodes
max_ranges
frontier_capacity
```

Если exact budget закончился для leaf, selector может использовать summary node.

## Output order

Из-за CUDA atomics порядок может отличаться от CPU DFS. Для attention это нормально: результат не зависит от порядка, если candidate set эквивалентен, потому что softmax считается по множеству кандидатов.
