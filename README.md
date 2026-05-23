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

# Decorator mode
@ops_dump("/path/to/dumps")
def run_model(input):
    return model(input)
```

## Comparing Two Dumps

```python
from acc import ops_comp

ops_comp("/path/to/dump_a", "/path/to/dump_b")
```

Prints LCS-matched operator sequence, then detailed per-operator comparison of inputs, kwargs, and outputs.

## License

MIT
