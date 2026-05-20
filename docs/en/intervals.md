# Intervals and Range Attention

## Semantic KV order

After tree construction, tokens are reordered so each subtree covers a contiguous interval in KV memory.

```text
node interval = [kv_start, kv_end)
```

This lets the attention kernel read exact tokens as ranges instead of lists of scattered token ids.

## Selector output

For a batch of query rows, the selector returns:

```text
range_starts   [Q, max_ranges]
range_ends     [Q, max_ranges]
summary_ids    [Q, max_summaries]
range_counts   [Q]
summary_counts [Q]
```

Each exact range means:

```text
attend to every token K/V in [start, end)
```

Each summary id means:

```text
attend to one summary K/V pseudo-token for that node
```

## Candidate set

For one query row:

```text
exact candidates   = all tokens from selected intervals
summary candidates = one pseudo-token per selected summary node
candidate_count    = exact_count + summary_count
```

The CUDA range attention kernel computes standard softmax over this candidate set.

## No subtree-size correction

Summary nodes are not multiplied by subtree size.

The current semantics are:

```text
summary node -> one key, one value, one softmax position
```

There is no `log(range_size)` correction and no expansion over the summarized interval.

## Per-row candidate limit

The kernel receives `candidate_counts[Q]`. This lets each row skip chunks beyond its real candidate count while keeping the launch shape static.
