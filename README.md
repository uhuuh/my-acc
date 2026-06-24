# PyTorch Operator Dump Tool

A tool for capturing and comparing PyTorch operator calls with input/output tensors.

## Dump Ops Data

```bash
pip install -e .
```

```python
from acc import ops_dump

# Context manager mode
with ops_dump("/path/to/dumps"):
    model(input)
```

### Backends

Two capture backends are available, configured via `capturer_backends`:

```python
from acc import ops_dump

# Default: ops only
with ops_dump("/path/to/dumps"):
    model(input)

# Ops + module forward hooks
with ops_dump("/path/to/dumps", model=model):
    model(input)
```

- `ops` (default) — captures every PyTorch operator via `TorchDispatchMode`
- `module` (default) — captures module forward calls via `register_forward_hook`; requires passing `model=`

Control via config or env var:

```python
from acc.config import config
config.update(capturer_backends="ops")         # ops only
# or: ACC_CAPTURER_BACKENDS=ops,module
```

## Comparing Two Dumps

```python
from acc import ops_comp

ops_comp("/path/to/dump_a", "/path/to/dump_b")
```

Prints LCS-matched operator sequence, then detailed per-operator comparison of inputs, kwargs, and outputs.

### Custom LCS key function

Control how operators are matched by passing `key_fn`:

```python
from acc import ops_comp

# Match only by operator/module key (ignore filename/capturer)
ops_comp(a, b, key_fn=lambda is_left, r: r.key)

# Custom: match by filename + key
ops_comp(a, b, key_fn=lambda is_left, r: f"{r.filename}:{r.key}")
```

`key_fn(is_left: bool, record: OperatorRecord) -> str` takes a boolean indicating which side (`True` for A, `False` for B) and the record, returns the matching key. Default prepends the capturer type: `f"{r.capturer}:{r.filename}::{r.key}"`.

## License

MIT
