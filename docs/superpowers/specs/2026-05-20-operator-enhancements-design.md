# PyTorch Operator Dump & Comparison Tool Enhancements Design

**Date:** 2026-05-20
**Status:** Design Approved

## Overview

Enhancements to existing PyTorch operator dump and comparison tool to address:
1. Scalar tensor comparison errors
2. Fixed float32 dtype in comparison results
3. Tensor dtype mismatch comparison support
4. Session directory naming with rank
5. Dump file naming with function info and clear separators
6. JSON/PKL split for dump files
7. Function info in comparison output
8. Generic type handling for unsupported types

## Architecture Changes

```
acc/
├── operator_dumper.py    # Session dir naming, dump file naming, JSON/PKL split
├── serialization.py      # New JSON/PKL serialization helpers
├── comparison_utils.py   # Tensor fixes, UnsupportedComparator enhancement
├── compare_dumps.py      # Add function to comparison header
└── __init__.py           # No changes
```

## Enhancement Details

### 1. Session Directory Naming with Rank

**Format:** `rank-pid-date-id`

- `rank`: `torch.distributed.get_rank()` if initialized, else `"None"` (literal string)
- `pid`: `os.getpid()`
- `date`: `YYYY-MM-DD_HH-MM-SS` (human-readable format)
- `id`: 8-char random hex from `uuid.uuid4().hex[:8]`

**Examples:**
- Distributed initialized: `0-12345-2026-05-20_14-30-30-a1b2c3d4`
- Not initialized: `None-12345-2026-05-20_14-30-30-a1b2c3d4`

**Implementation in `operator_dumper.py`:**
```python
import uuid
import torch.distributed as dist

def __enter__(self):
    # Get rank
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = "None"
    
    pid = os.getpid()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    session_id = uuid.uuid4().hex[:8]
    
    self.session_dir = os.path.join(
        self.dump_path, 
        f"{rank}-{pid}-{timestamp}-{session_id}"
    )
    
    os.makedirs(self.session_dir, exist_ok=False)
```

### 2. Dump File Naming with Function Info

**Pattern:** `{seq:06d}__{file}__{func}__{op}.{ext}`

- `seq`: 6-digit zero-padded sequence number
- `file`: source filename (e.g., `transformer.py`)
- `func`: function name only (e.g., `forward`)
- `op`: operator name sanitized
- `ext`: `.json` or `.pkl`
- Separator: `__` (double underscore)

**Global Function Handling:**
- If global function (no module context): file set to `<global>`
- Pattern: `{seq:06d}__<global>__{func}__{op}.{ext}`

**Implementation in `operator_dumper.py`:**
```python
def _dump_operation(self, func, args, kwargs, result):
    stack = traceback.extract_stack()
    
    # Find caller frame
    filename = "<global>"
    func_name = ""
    lineno = 0
    
    for frame_info in reversed(stack):
        if not frame_info.filename.endswith('operator_dumper.py'):
            filename = os.path.basename(frame_info.filename)
            func_name = frame_info.name
            lineno = frame_info.lineno
            break
    
    # Sanitize for filename
    filename_safe = filename.replace('.py', '').replace('/', '_').replace('\\', '_')
    opname_safe = str(func).replace('.', '_').replace('::', '_')
    
    # Generate IDs for tensors
    tensor_ids = {}
    inputs_with_ids = self._add_tensor_ids(args, tensor_ids)
    
    # JSON file
    json_data = {
        'sequence': self.sequence,
        'filename': filename,
        'function': func_name,
        'lineno': lineno,
        'opname': str(func),
        'call_stack': ''.join(traceback.format_stack()),
        'inputs': inputs_with_ids
    }
    
    json_filename = f"{self.sequence:06d}__{filename_safe}__{func_name}__{opname_safe}.json"
    pkl_filename = f"{self.sequence:06d}__{filename_safe}__{func_name}__{opname_safe}.pkl"
```

### 3. JSON/PKL Split for Dump Files

**JSON File Content:**
```json
{
  "sequence": 1,
  "filename": "transformer.py",
  "function": "forward",
  "lineno": 42,
  "opname": "torch.add",
  "call_stack": "Traceback (most recent call last):\n  File ...",
  "inputs": [
    {"type": "tensor", "id": "a1b2c3d4"},
    {"type": "int", "id": "e5f6g7h8"},
    {"type": "float", "id": "i9j0k1l2"}
  ]
}
```

**PKL File Content:**
```python
{
    "a1b2c3d4": tensor_cpu_data,
    "e5f6g7h8": 42,
    "i9j0k1l2": 3.14,
    "m3n4o5p6": str_data,
    "q7r8s9t0": None
}
```

**Data Storage Rules:**
- JSON: only `type` and `id` fields for each input
- PKL: stores all actual data with ID as key
- 8-char random hex ID per value: `uuid.uuid4().hex[:8]`

**Implementation in `serialization.py`:**
```python
import uuid
import json
import pickle

def _generate_id():
    return uuid.uuid4().hex[:8]

def _serialize_to_json_pkl(obj):
    """
    Split object into JSON metadata (type + id) and PKL data.
    
    Returns:
        Tuple of (json_data, pkl_data)
    """
    pkl_data = {}
    
    def process_value(value):
        id = _generate_id()
        
        if isinstance(value, torch.Tensor):
            pkl_data[id] = value.detach().cpu()
            return {'type': 'tensor', 'id': id}
        elif isinstance(value, np.ndarray):
            pkl_data[id] = value
            return {'type': 'numpy', 'id': id}
        elif isinstance(value, int):
            pkl_data[id] = value
            return {'type': 'int', 'id': id}
        elif isinstance(value, float):
            pkl_data[id] = value
            return {'type': 'float', 'id': id}
        elif isinstance(value, str):
            pkl_data[id] = value
            return {'type': 'str', 'id': id}
        elif value is None:
            pkl_data[id] = None
            return {'type': 'None', 'id': id}
        elif isinstance(value, (list, tuple)):
            pkl_data[id] = value
            return {'type': type(value).__name__, 'id': id}
        elif isinstance(value, dict):
            pkl_data[id] = value
            return {'type': 'dict', 'id': id}
        else:
            pkl_data[id] = value
            return {'type': 'generic', 'id': id}
    
    json_data = process_value(obj)
    return json_data, pkl_data
```

### 4. Scalar Tensor Handling

**Problem:**
- 0-dim tensors (scalar) cause comparison errors when checking shape

**Fix:**
- Before comparison: check `tensor.ndim == 0`
- If scalar: `tensor = tensor.unsqueeze(0)` to add dimension
- After comparison: restore original dtype info

**Implementation in `comparison_utils.py`:**
```python
class TensorComparator(ElementComparator):
    def compare(self) -> Dict:
        a = torch.from_numpy(self.a) if isinstance(self.a, np.ndarray) else self.a
        b = torch.from_numpy(self.b) if isinstance(self.b, np.ndarray) else self.b
        
        # Store original dtype
        dtype_a_original = a.dtype
        dtype_b_original = b.dtype
        
        dtype_match = a.dtype == b.dtype
        shape_match = a.shape == b.shape
        
        # Handle scalar tensors
        if a.ndim == 0:
            a = a.unsqueeze(0)
        if b.ndim == 0:
            b = b.unsqueeze(0)
        
        if not shape_match:
            return {
                'dtype_match': dtype_match,
                'shape_match': False,
                'dtype_original_a': str(dtype_a_original),
                'dtype_original_b': str(dtype_b_original),
                'content_skipped': True
            }
        
        # Cast to float32 for comparison
        a_float = a.float()
        b_float = b.float()
        
        exact_match = torch.allclose(a_float, b_float, rtol=0, atol=0)
        ...
        
        return {
            'dtype_match': dtype_match,
            'shape_match': shape_match,
            'dtype_original_a': str(dtype_a_original),
            'dtype_original_b': str(dtype_b_original),
            'exact_match': exact_match,
            ...
        }
```

### 5. Different Dtype Tensor Comparison

**Requirement:**
- Compare tensors with different dtypes
- Cast both to float32 for content comparison
- Return original dtype info

**Implementation:**
- Always cast to float32 before content comparison
- Preserve original dtypes in result dict
- Type info shows both original dtypes

```python
def get_type_info(self) -> Tuple[str, str]:
    dtype_a = str(self.a.dtype)
    shape_a = list(self.a.shape)
    dtype_b = str(self.b.dtype)
    shape_b = list(self.b.shape)
    return f"tensor(dtype={dtype_a}, shape={shape_a})", f"tensor(dtype={dtype_b}, shape={shape_b})"
```

### 6. UnsupportedComparator Enhancement

**Old Behavior:**
- Mark as unsupported, no comparison

**New Behavior:**
- Description: `str(obj)`
- Comparison: `a == b` using Python equality
- Result: `exact_match`, `str_a`, `str_b`

**Implementation in `comparison_utils.py`:**
```python
class UnsupportedComparator(ElementComparator):
    def get_type_info(self) -> Tuple[str, str]:
        return str(self.a), str(self.b)
    
    def compare(self) -> Dict:
        try:
            exact_match = self.a == self.b
        except Exception:
            exact_match = False
        
        return {
            'exact_match': exact_match,
            'str_a': str(self.a),
            'str_b': str(self.b)
        }
```

**Example Output:**
```
Inputs[0] | "list([1, 2, 3])" | "list([1, 2, 4])" | exact_match=False
```

### 7. Function Info in Comparison Output

**LCS Signature Update:**
- Old: `{filename}::{opname}`
- New: `{filename}::{function}::{opname}`

**Comparison Header Format:**
```
[COMPARE] A:{seq:06d}__{file}__{func}__{op} <-> B:{seq:06d}__{file}__{func}__{op}:
```

**Example:**
```
[COMPARE] A:000001__transformer__forward__torch_add <-> B:000002__transformer__forward__torch_add:
  Inputs[0] | tensor(dtype=float32, shape=[32,64]) | tensor(dtype=float32, shape=[32,64]) | exact_match=True
```

**MATCH Log:**
```
[MATCH] A:000001__transformer__forward__torch_add <-> B:000002__transformer__forward__torch_add
```

**SKIP Log:**
```
[SKIP] A:000003__transformer__forward__torch_dropout (no match in B)
```

**Implementation in `compare_dumps.py`:**
```python
def ops_comp(dump_dir_a: str, dump_dir_b: str):
    # Load JSON files for metadata
    dumps_a = _load_dumps(dump_dir_a)
    dumps_b = _load_dumps(dump_dir_b)
    
    # Build signatures with function
    sigs_a = [f"{d['filename']}::{d['function']}::{d['opname']}" for d in dumps_a]
    sigs_b = [f"{d['filename']}::{d['function']}::{d['opname']}" for d in dumps_b]
    
    # Format comparison header
    for idx_a, idx_b in matched_pairs:
        dump_a = dumps_a[idx_a]
        dump_b = dumps_b[idx_b]
        
        header_a = f"{dump_a['sequence']:06d}__{dump_a['filename'].replace('.py', '')}__{dump_a['function']}__{dump_a['opname'].replace('.', '_').replace('::', '_')}"
        header_b = f"{dump_b['sequence']:06d}__{dump_b['filename'].replace('.py', '')}__{dump_b['function']}__{dump_b['opname'].replace('.', '_').replace('::', '_')}"
        
        print(f"[COMPARE] A:{header_a} <-> B:{header_b}:")
```

### 8. Loading JSON/PKL Split Files

**Implementation in `compare_dumps.py`:**
```python
def _load_dumps(dump_dir: str) -> List[Dict]:
    """
    Load all dump files from directory.
    Each dump has .json (metadata) and .pkl (actual data).
    """
    dumps = []
    
    for filename in os.listdir(dump_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(dump_dir, filename)
            pkl_path = json_path.replace('.json', '.pkl')
            
            # Load JSON metadata
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            # Load PKL actual data
            pkl_data = {}
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    pkl_data = pickle.load(f)
            
            # Reconstruct inputs with actual data
            inputs = [_reconstruct_values(inp, pkl_data) for inp in metadata['inputs']]
            
            dump_data = {
                'sequence': metadata['sequence'],
                'filename': metadata['filename'],
                'function': metadata['function'],
                'lineno': metadata.get('lineno', 0),
                'opname': metadata['opname'],
                'call_stack': metadata.get('call_stack', ''),
                'inputs': inputs
            }
            
            dumps.append(dump_data)
    
    dumps.sort(key=lambda x: x['sequence'])
    return dumps

def _reconstruct_values(json_data, pkl_data):
    """Reconstruct values from JSON metadata and PKL data."""
    id = json_data['id']
    return pkl_data[id]
```

## File Changes Summary

| File | Changes |
|------|---------|
| `operator_dumper.py` | Session dir naming with rank, JSON/PKL split dump, function extraction |
| `serialization.py` | New helpers for JSON/PKL serialization, ID generation, value reconstruction |
| `comparison_utils.py` | Scalar tensor fix, dtype mismatch comparison, UnsupportedComparator enhancement |
| `compare_dumps.py` | Load JSON/PKL split, function in comparison header, updated signature format |
| `__init__.py` | No changes |

## Error Handling

| Case | Handling |
|------|----------|
| Distributed not initialized | rank = `"None"` (string) |
| Scalar tensor comparison | Unsqueeze to add dimension |
| Dtype mismatch | Cast to float32, preserve original dtype info |
| Unpicklable objects | Use str() representation |

## Testing Considerations

1. Test scalar tensor comparison (0-dim)
2. Test different dtype tensor comparison
3. Test distributed session naming (rank=0, rank=None)
4. Test JSON/PKL reconstruction
5. Test function extraction from call stack
6. Test unsupported type comparison (list, dict, custom objects)

## Migration Notes

- Old dump files (single .pkl) will need migration script for comparison
- Session directory format change: old `{pid}_{timestamp}` → new `{rank}-{pid}-{timestamp}-{id}`
- Dump file format change: old single `.pkl` → new `.json` + `.pkl`