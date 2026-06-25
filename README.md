# PyTorch Operator Dump Tool

Capture and inspect PyTorch operator calls with input/output tensors.

## Install

```bash
pip install -e .
```

## acc_dump — Capture Operators

```python
from acc import acc_dump

with acc_dump("/path/to/dumps"):
    model(input)
```

### Backends

| Backend | Description |
|---|---|
| `ops` (default) | Captures every PyTorch operator via `TorchDispatchMode` |
| `module` (default) | Captures module forward calls via `register_forward_hook`; requires `model=` |

```python
# Ops + module forward hooks
with acc_dump("/path/to/dumps", model=model):
    model(input)

# Ops only
with acc_dump("/path/to/dumps", capturer_backends="ops"):
    model(input)
```

### Output files

Each captured operator produces two files:

- `<save_id>.json` — metadata (seq, file, func, lineno, call stack)
- `<save_id>.pkl` — serialized inputs/outputs as `CacheEntry` references

Unique tensors are deduplicated to `cache/<cache_id>.pt`.

## acc_info — Inspect a Dump

```python
from acc import acc_info

# Print all operator inputs/outputs
acc_info("/path/to/dumps")

# Filter: keep only conv2d ops
acc_info("/path/to/dumps", filter_fn=lambda r: "conv2d" not in r.key)
```

`filter_fn(record) -> bool` — return `True` to skip a record.

For each operator, prints kwargs and outputs. Tensors and numpy arrays show shape, dtype, max, min, mean, MSE, and quartiles.

## acc_comp — Compare Two Dumps

```python
from acc import acc_comp

acc_comp("/path/to/dump_a", "/path/to/dump_b")
```

Prints LCS-matched operator sequence with per-operator tensor comparison.

### Custom matching

```python
# Match only by operator key (ignore filename/capturer)
acc_comp(a, b, key_fn=lambda is_left, r: r.key)

# Filter: compare only conv2d ops
acc_comp(a, b, filter_fn=lambda is_left, r: "conv2d" not in r.key)
```

`key_fn(is_left: bool, record: OperatorRecord) -> str` — custom LCS matching key. Default: `f"{r.capturer}:{r.filename}::{r.key}"`.

`filter_fn(is_left: bool, record: OperatorRecord) -> bool` — return `True` to exclude a record from comparison.

## License

MIT
