# Интервалы и Range Attention

## Semantic KV order

После построения дерева токены переупорядочиваются так, чтобы каждое поддерево покрывало contiguous interval в KV memory.

```text
node interval = [kv_start, kv_end)
```

Так attention kernel читает exact tokens как ranges, а не как список scattered token ids.

## Output selector-а

Для batch query rows selector возвращает:

```text
range_starts   [Q, max_ranges]
range_ends     [Q, max_ranges]
summary_ids    [Q, max_summaries]
range_counts   [Q]
summary_counts [Q]
```

Каждый exact range означает:

```text
attend to every token K/V in [start, end)
```

Каждый summary id означает:

```text
attend to one summary K/V pseudo-token for that node
```

## Candidate set

Для одной query row:

```text
exact candidates   = все токены из selected intervals
summary candidates = один pseudo-token на каждый selected summary node
candidate_count    = exact_count + summary_count
```

CUDA range attention kernel считает обычный softmax по этому candidate set.

## Без поправки на размер поддерева

Summary nodes не умножаются на размер поддерева.

Текущая семантика:

```text
summary node -> one key, one value, one softmax position
```

Нет `log(range_size)` correction и нет expansion по summarized interval.

## Per-row candidate limit

Kernel получает `candidate_counts[Q]`. Поэтому каждая row может пропускать chunks за пределами своего реального числа кандидатов, сохраняя static launch shape.
