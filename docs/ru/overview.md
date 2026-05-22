# Обзор Tree Attention

## Цель

Tree Attention снижает стоимость long-context decode: вместо full attention по всем cached tokens он считает attention по:

- exact token ranges, выбранным поиском по дереву;
- summary spans, где один score представляет много реальных KV токенов.

Dense decode attention:

```text
scores = q @ K_all.T * scale
weights = softmax(scores)
out = weights @ V_all
```

Score-only Tree Attention:

```text
exact candidates:   score_i = q @ K[i] * scale, value_i = V[i]
summary candidate:  score_s = q @ node_key * scale + log(end - start)
                    value_s = mean(V[start:end])
weights = softmax(compact_scores)
out = weights @ compact_values
```

`log(end - start)` делает один summary candidate эквивалентным тому, что один и тот же score применили ко всем реальным токенам внутри span:

```text
softmax(summary_score + log(length)) * mean_v
= exp(summary_score) * V[start:end].sum / denom
```

## Pipeline

1. Строим semantic tree над KV-cache.
2. Храним каждое поддерево как contiguous KV interval.
3. Для каждого query запускаем range-bound tree search.
4. Получаем exact token intervals и summary spans.
5. Упаковываем compact candidates на CUDA.
6. Считаем softmax и value aggregation только по compact candidates.

## Текущий prototype

Лучший текущий путь — v69 compact score-pack:

- selector layer: CUDA range-bound traversal;
- score layer: exact token scores плюс summary-span scores;
- value layer: exact `V` плюс prefix sums по real `V` для summary spans;
- compact attention layer: softmax только по physical candidates.

Synthetic Kaggle/T4 decode results показывают выигрыш только на длинном контексте:

| seq len | dense ms | tree ms | speedup | output cosine |
|---:|---:|---:|---:|---:|
| 64k | 2.240 | 2.716 | 0.825x | 0.98617 |
| 128k | 3.812 | 3.361 | 1.134x | 0.97960 |
| 256k | 6.612 | 5.138 | 1.287x | 0.98357 |
| 512k | 8.090 | 5.569 | 1.453x | 0.98340 |

## Главный tradeoff

Tree Attention полезен только если physical candidate count сильно меньше full context и качество модели остаётся приемлемым.

```text
Dense decode cost: O(context_tokens)
Tree decode cost:  O(visited_nodes + exact_tokens + summary_spans)
Build cost:        amortized over future decode steps
```

Текущий prototype ещё не production-ready. Следующий обязательный шаг — проверка качества на реальной модели: perplexity/logit deltas, generation stability, retrieval и long-context QA.
