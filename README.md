# PyTorch Operator Dump Tool

A tool for capturing and comparing PyTorch operator calls with input/output tensors.

## Features

- **Operator Dump**: Capture all PyTorch operator calls with metadata (call stack, inputs, outputs)
- **Dump Comparison**: Compare two dump sessions using LCS algorithm to find differences
- **Global Control**: Enable/disable dump behavior globally
- **Custom Operator Support**: Automatic patch for custom operator registration

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

### Global Control

```python
from acc import set_dump_enabled, get_dump_enabled

# Disable all dump operations
set_dump_enabled(False)

# Enable dump operations
set_dump_enabled(True)

# Check current status
print(get_dump_enabled())
```

### Direct Access to Dumper Manager

```python
from acc import dumper_manager

# Access session state
print(dumper_manager.session_dir)
print(dumper_manager.sequence)
print(dumper_manager.active)
```

## Output Format

Each operator call generates two files:

- `*.json`: Metadata (sequence, filename, function, lineno, opname, call_stack)
- `*.pkl`: Tensor data (inputs and outputs)

File naming: `{sequence:06d}__{filename}__{function}__{opname}.json`

## Special Logs

- `[DUMP PATCH]`: Logged when torch.library.impl is patched for custom operator capture
- `[DUMP WARN]`: Logged when nested ops_dump is detected (ignored, continues with existing session)

## Custom Operators

### Dispatch Key Behavior

| Dispatch Key | Internal Operators Captured |
|--------------|----------------------------|
| CompositeImplicitAutograd | ✅ Yes (decomposition) |
| CompositeExplicitAutograd | ❌ No (whole operator) |
| CPU/CUDA | ❌ No (backend implementation) |

For custom operators registered with `CompositeExplicitAutograd` or backend keys, internal operators are not captured individually. Use `CompositeImplicitAutograd` if you need decomposition-based capture.

## API Reference

### `ops_dump(dump_path: str, max_tensor_size_mb: int = 10240)`

Context manager for capturing operator calls.

### `ops_comp(dump_dir_a: str, dump_dir_b: str)`

Compare two dump sessions.

### `set_dump_enabled(enabled: bool)`

Set global dump enabled flag.

### `get_dump_enabled() -> bool`

Get global dump enabled flag.

### `DUMP_ENABLED`

Module-level boolean for controlling dump behavior.

### `dumper_manager`

Global singleton for dump state management.

## License

MIT