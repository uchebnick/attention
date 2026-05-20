# Обзор Tree Attention

## Цель

Tree Attention снижает стоимость decode на длинном контексте: вместо full attention по всем токенам он считает attention по выбранным exact ranges и summary nodes.

Dense attention:

```text
scores = q @ K_all.T * scale
weights = softmax(scores)
out = weights @ V_all
```

Tree Attention:

```text
selected_K = concat(exact token K, summary node K)
selected_V = concat(exact token V, summary node V)
scores = q @ selected_K.T * scale
weights = softmax(scores)
out = weights @ selected_V
```

Summary node — это один pseudo-token. Он не разворачивается в range и не умножается на размер поддерева.

## Pipeline

1. Строим semantic tree над KV-cache.
2. Храним каждое поддерево как contiguous KV interval.
3. Для каждого query запускаем range-bound поиск по дереву.
4. Получаем exact token intervals и summary node ids.
5. Запускаем range attention по сжатому candidate set.

## Главный tradeoff

Метод быстрее только если выбранных кандидатов сильно меньше, чем полный контекст.

```text
Dense cost: O(context_tokens)
Tree decode cost: O(visited_nodes + selected_tokens + summary_tokens)
Build cost: amortized over future queries
```

## Текущий prototype

Реализация разделена на два слоя:

- selector layer: выбирает exact intervals и summary nodes;
- range attention layer: считает обычный softmax attention по этим кандидатам.

Selector может работать на CPU или CUDA. Range attention имеет CUDA kernel.
