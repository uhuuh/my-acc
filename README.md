# PyTorch Operator Dump Tool

A tool for capturing and comparing PyTorch operator calls with input/output tensors.

## Features

- **Operator Dump**: Capture all PyTorch operator calls with metadata (call stack, inputs, outputs)
- **Dump Comparison**: Compare two dump sessions using LCS algorithm to find differences
- **Content-Addressable Cache**: Deduplicate tensor/numpy data using BLAKE2 hashes
- **Global Control**: Enable/disable dump behavior via environment variable
- **Custom Operator Support**: Automatic patch for custom operator registration
- **ETA Progress**: Real-time ETA during comparison operations

## Installation

```bash
pip install -e .
```

## Usage

### Basic Dump

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

### Compare Dumps

```python
from acc import ops_comp

ops_comp("/path/to/dump_session_a", "/path/to/dump_session_b")
```

### Global Control via Environment Variable

```bash
# Disable all dump operations
export ACC_DUMP_ENABLED=0

# Enable dump operations (default)
export ACC_DUMP_ENABLED=1
```

## Output Format

Each operator call generates two files:

- `*.json`: Metadata (sequence, filename, function, lineno, opname, call_stack)
- `*.pkl`: Tensor data (inputs as args/kwargs, outputs)

Additionally, a `storage/` subdirectory contains cached tensor/numpy data files:
- `storage/{cache_id}.pkl`: Cached tensor/numpy data (deduplicated by content hash)

File naming: `{sequence:06d}__{filename}__{function}__{opname}.json`

## Special Logs

- `[DUMP PATCH]`: Logged when torch.library.impl is patched for custom operator capture
- `[DUMP WARN]`: Logged when tensor size exceeds limit or serialization fails
- `[COMPARE X/Y | ETA: ...]`: Progress indicator during comparison

## Custom Operators

### Dispatch Key Behavior

| Dispatch Key | Internal Operators Captured |
|--------------|----------------------------|
| CompositeImplicitAutograd | Yes (decomposition) |
| CompositeExplicitAutograd | No (whole operator) |
| CPU/CUDA | No (backend implementation) |

For custom operators registered with `CompositeExplicitAutograd` or backend keys, internal operators are not captured individually. Use `CompositeImplicitAutograd` if you need decomposition-based capture.

## API Reference

### `ops_dump(dump_path: str, max_tensor_size_mb: int = 10240, enable_cache: bool = True)`

Context manager for capturing operator calls.

- `dump_path`: Directory to store dump files
- `max_tensor_size_mb`: Maximum tensor size to serialize (default: 10240 MB)
- `enable_cache`: Enable content-addressable cache for tensor deduplication (default: True)

### `ops_comp(dump_dir_a: str, dump_dir_b: str)`

Compare two dump sessions using LCS matching.

### `SerializationSession`

Manages a single serialization session, integrating cache storage.

```python
from acc import SerializationSession

# Load metadata (no tensor data)
record = SerializationSession.load_metadata("path/to/file.json")

# Load data (with cache resolution)
inputs, outputs = SerializationSession.load_data("path/to/file.pkl", "path/to/storage")
```

### `OperatorRecord`

Data structure for a single operator dump, with separated args and kwargs.

```python
from acc import OperatorRecord

# record.args    - list of positional arguments
# record.kwargs  - dict of keyword arguments
# record.outputs - list of outputs
# record.sequence, record.filename, record.opname, etc.
```

## License

MIT