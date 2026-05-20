# Устройство дерева

## Что хранит node

Каждый node хранит:

```text
id
key
value
radius
children
kv_start
kv_end
```

Смысл полей:

- `key`: representative key для routing и summary attention;
- `value`: representative value, если поддерево заменяется summary;
- `radius`: bound на расстояние ключей поддерева от key узла;
- `children`: ids дочерних узлов;
- `kv_start`, `kv_end`: contiguous interval, покрываемый узлом.

## Leaves и internal nodes

Leaves содержат реальные token ids. Internal nodes содержат summaries своих children.

```text
leaf.value     = summary real token values
internal.value = summary child values
leaf.key       = summary real token keys
internal.key   = summary child keys
```

## Multi-root layout

Cache может иметь несколько top-level roots. Это делает поиск более параллельным и убирает bottleneck одного большого root.

```text
root_ids = [root_0, root_1, ..., root_n]
```

Virtual root может существовать для bookkeeping, но CUDA selection стартует с top-level roots.

## Flat GPU metadata

CUDA selector не ходит по Python objects. Он использует flat tensors:

```text
node_keys       [node_count, dim]
node_values     [node_count, value_dim]
node_radii      [node_count]
node_kv_starts  [node_count]
node_kv_ends    [node_count]
root_ids        [root_count]
is_leaf         [node_count]
```

Children хранятся через CSR:

```text
child_offsets [node_count + 1]
child_indices [edge_count]

children(node) = child_indices[child_offsets[node] : child_offsets[node + 1]]
```
