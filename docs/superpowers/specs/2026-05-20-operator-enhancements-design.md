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
    import json
    import traceback
    
    stack = traceback.extract_stack()
    
    filepath = ""
    filename = "<global>"
    func_name = ""
    lineno = 0
    
    for frame_info in reversed(stack):
        if not frame_info.filename.endswith('operator_dumper.py'):
            filepath = frame_info.filename
            filename = os.path.basename(frame_info.filename)
            func_name = frame_info.name
            lineno = frame_info.lineno
            break
    
    call_stack = ''.join(traceback.format_stack())
    
    filename_safe = filename.replace('.py', '').replace('/', '_').replace('\\', '_')
    opname_safe = str(func).replace('.', '_').replace('::', '_')
    
    data_list = _serialize_inputs(args, kwargs)
    
    json_data = {
        'sequence': self.sequence,
        'filepath': filepath,
        'filename': filename,
        'function': func_name,
        'lineno': lineno,
        'opname': str(func),
        'call_stack': call_stack
    }
    
    json_filename = f"{self.sequence:06d}__{filename_safe}__{func_name}__{opname_safe}.json"
    pkl_filename = f"{self.sequence:06d}__{filename_safe}__{func_name}__{opname_safe}.pkl"
    
    json_path = os.path.join(self.session_dir, json_filename)
    pkl_path = os.path.join(self.session_dir, pkl_filename)
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data_list, f)
    
    print(f"[DUMP] {self.sequence:06d} | {filename}:{lineno} | {func} | saved to {json_filename}")
    self.sequence += 1
```

### 3. JSON/PKL Split for Dump Files

**JSON File Content:**
```json
{
  "sequence": 1,
  "filepath": "/path/to/models/transformer.py",
  "filename": "transformer.py",
  "function": "forward",
  "lineno": 42,
  "opname": "torch.add",
  "call_stack": "Traceback (most recent call last):\n  File ...\n    ...\n"
}
```

**PKL File Content:**
```python
[
    tensor_cpu_data,
    42,
    3.14
]
```

**Data Storage Rules:**
- JSON: only metadata (no input information)
- PKL: direct list of all input data

**Implementation in `serialization.py`:**
```python
def _serialize_inputs(args, kwargs):
    """
    Serialize inputs to list for PKL storage.
    
    Returns:
        List of input data (tensors on CPU)
    """
    import torch
    import numpy as np
    
    data_list = []
    
    for arg in args:
        if isinstance(arg, torch.Tensor):
            data_list.append(arg.detach().cpu())
        elif isinstance(arg, numpy.ndarray):
            data_list.append(arg)
        else:
            data_list.append(arg)
    
    if kwargs:
        data_list.append(kwargs)
    
    return data_list
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
[COMPARE] {dump_filename}
```

**Example:**
```
[COMPARE] 000001__transformer__forward__torch_add.json
  Inputs[0] | tensor(dtype=float32, shape=[32,64]) | tensor(dtype=float32, shape=[32,64]) | exact_match=True
```

**MATCH Log:**
```
[MATCH] transformer.py(torch.add) <-> transformer.py(torch.add)
```

**SKIP Log:**
```
[SKIP] transformer.py(torch.dropout) (no match in B)
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
    
    # Match and skip logs
    for i, dump in enumerate(dumps_a):
        if i in matched_map_a:
            match_idx = matched_map_a[i]
            key_a = f"{dump['filename']}({dump['opname']})"
            key_b = f"{dumps_b[match_idx]['filename']}({dumps_b[match_idx]['opname']})"
            print(f"[MATCH] {key_a} <-> {key_b}")
        else:
            key = f"{dump['filename']}({dump['opname']})"
            print(f"[SKIP] {key} (no match in B)")
    
    for j, dump in enumerate(dumps_b):
        if j not in matched_map_b:
            key = f"{dump['filename']}({dump['opname']})"
            print(f"[SKIP] {key} (no match in A)")
    
    # Comparison
    for idx_a, idx_b in matched_pairs:
        dump_a = dumps_a[idx_a]
        dump_b = dumps_b[idx_b]
        
        filename_safe = dump_a['filename'].replace('.py', '').replace('/', '_').replace('\\', '_')
        func_name = dump_a['function']
        opname_safe = dump_a['opname'].replace('.', '_').replace('::', '_')
        
        dump_filename = f"{dump_a['sequence']:06d}__{filename_safe}__{func_name}__{opname_safe}.json"
        
        print(f"[COMPARE] {dump_filename}")
        
        # Compare inputs
        inputs_a = dump_a['inputs']
        inputs_b = dump_b['inputs']
        
        max_inputs = max(len(inputs_a), len(inputs_b))
        
        for i in range(max_inputs):
            if i >= len(inputs_a):
                comparator = MissingInAComparator(inputs_b[i])
            elif i >= len(inputs_b):
                comparator = MissingInBComparator(inputs_a[i])
            else:
                comparator = create_comparator(inputs_a[i], inputs_b[i])
            
            left_info, right_info = comparator.get_type_info()
            result = comparator.compare()
            log = format_comparison_log(result)
            print(f"  Inputs[{i}] | {left_info} | {right_info} | {log}")
```

### 8. Loading JSON/PKL Split Files

**Implementation in `compare_dumps.py`:**
```python
def _load_dumps(dump_dir: str) -> List[Dict]:
    """
    Load all dump files from directory.
    Each dump has .json (metadata) and .pkl (input data list).
    """
    dumps = []
    
    for filename in os.listdir(dump_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(dump_dir, filename)
            pkl_path = json_path.replace('.json', '.pkl')
            
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            inputs = []
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    inputs = pickle.load(f)
            
            dump_data = {
                'sequence': metadata['sequence'],
                'filepath': metadata.get('filepath', ''),
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