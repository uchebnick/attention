# Tree Structure

## Node contents

Each tree node stores:

```text
id
key
value
radius
children
kv_start
kv_end
```

Field meanings:

- `key`: representative key used for routing and summary attention;
- `value`: representative value used when the subtree is summarized;
- `radius`: bound on how far subtree keys can be from the node key;
- `children`: child node ids;
- `kv_start`, `kv_end`: contiguous interval covered by the node.

## Leaves and internal nodes

Leaves contain real token ids. Internal nodes contain summaries of their children.

```text
leaf.value     = summary of real token values
internal.value = summary of child values
leaf.key       = summary of real token keys
internal.key   = summary of child keys
```

## Multi-root layout

The cache can use several top-level roots. This makes search more parallel and avoids one very large root bottleneck.

```text
root_ids = [root_0, root_1, ..., root_n]
```

A virtual root may exist for bookkeeping, but CUDA selection starts from the top-level roots.

## Flat GPU metadata

The CUDA selector does not traverse Python objects. It uses flat tensors:

```text
node_keys       [node_count, dim]
node_values     [node_count, value_dim]
node_radii      [node_count]
node_kv_starts  [node_count]
node_kv_ends    [node_count]
root_ids        [root_count]
is_leaf         [node_count]
```

Children are stored as CSR:

```text
child_offsets [node_count + 1]
child_indices [edge_count]

children(node) = child_indices[child_offsets[node] : child_offsets[node + 1]]
```
