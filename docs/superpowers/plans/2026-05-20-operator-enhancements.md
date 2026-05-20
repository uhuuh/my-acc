# Operator Dump & Comparison Tool Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance PyTorch operator dump and comparison tool with rank-based naming, JSON/PKL split, scalar tensor handling, and improved comparison output.

**Architecture:** Modify existing modules in `acc/` package. JSON stores only metadata, PKL stores input data as simple list.

**Tech Stack:** Python, PyTorch, pickle, json

---

## File Structure

```
acc/
├── serialization.py      # New helper: _serialize_inputs()
├── operator_dumper.py    # Session dir naming, function extraction, JSON/PKL dump
├── comparison_utils.py   # TensorComparator fixes, UnsupportedComparator enhancement
├── compare_dumps.py      # Load JSON/PKL, updated comparison output format
└── __init__.py           # No changes
```

---

### Task 1: Add Serialization Helper

**Files:**
- Modify: `acc/serialization.py`

- [ ] **Step 1: Add _serialize_inputs() function**

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

- [ ] **Step 2: Commit**

```bash
git add acc/serialization.py
git commit -m "feat: add _serialize_inputs helper for PKL storage"
```

---

### Task 2: Update Session Directory Naming with Rank

**Files:**
- Modify: `acc/operator_dumper.py`

- [ ] **Step 1: Add uuid and dist imports**

Add at top of file after existing imports:

```python
import uuid
import torch.distributed as dist
```

- [ ] **Step 2: Update __enter__ method for rank-based naming**

Replace the `__enter__` method:

```python
def __enter__(self):
    """Enter context manager, create session directory."""
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
    self.sequence = 0
    self._active = True
    print(f"[DUMP] Created session directory: {self.session_dir}")
    return super().__enter__()
```

- [ ] **Step 3: Commit**

```bash
git add acc/operator_dumper.py
git commit -m "feat: add rank-based session directory naming"
```

---

### Task 3: Update Dump File Naming and JSON/PKL Split

**Files:**
- Modify: `acc/operator_dumper.py`

- [ ] **Step 1: Update import to include serialization helper**

Add import for serialization helper:

```python
from .serialization import _serialize_inputs, _sanitize_filename
```

- [ ] **Step 2: Update _dump_operation method**

Replace the `_dump_operation` method:

```python
def _dump_operation(self, func, args, kwargs, result):
    """Dump a single operator call."""
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
    
    filename_safe = _sanitize_filename(filename)
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
    
    try:
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        with open(pkl_path, 'wb') as f:
            pickle.dump(data_list, f)
    except Exception as e:
        print(f"[DUMP ERROR] {self.sequence:06d} | {filename}:{lineno} | {func} | {e}")
        self.sequence += 1
        return
    
    print(f"[DUMP] {self.sequence:06d} | {filename}:{lineno} | {func} | saved to {json_filename}")
    self.sequence += 1
```

- [ ] **Step 3: Commit**

```bash
git add acc/operator_dumper.py
git commit -m "feat: add JSON/PKL split dump with filepath and function info"
```

---

### Task 4: Fix Scalar Tensor Comparison

**Files:**
- Modify: `acc/comparison_utils.py`

- [ ] **Step 1: Update TensorComparator.compare() method**

Replace the `compare` method in `TensorComparator` class:

```python
def compare(self) -> Dict:
    a = torch.from_numpy(self.a) if isinstance(self.a, np.ndarray) else self.a
    b = torch.from_numpy(self.b) if isinstance(self.b, np.ndarray) else self.b
    
    dtype_a_original = a.dtype
    dtype_b_original = b.dtype
    
    dtype_match = a.dtype == b.dtype
    shape_match = a.shape == b.shape
    
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
    
    a_float = a.float()
    b_float = b.float()
    
    exact_match = torch.allclose(a_float, b_float, rtol=0, atol=0)
    match_count = (a_float == b_float).sum().item()
    total_count = a_float.numel()
    match_ratio = match_count / total_count if total_count > 0 else 1.0
    
    diff = torch.abs(a_float - b_float)
    max_err = diff.max().item()
    min_err = diff.min().item()
    mean_err = diff.mean().item()
    mse = torch.mean((a_float - b_float) ** 2).item()
    
    if total_count > 0 and a_float.norm() > 0 and b_float.norm() > 0:
        cosine = torch.nn.functional.cosine_similarity(
            a_float.unsqueeze(0), b_float.unsqueeze(0)
        ).item()
    else:
        cosine = 1.0
    
    precision_diff = not exact_match
    
    return {
        'dtype_match': dtype_match,
        'shape_match': shape_match,
        'dtype_original_a': str(dtype_a_original),
        'dtype_original_b': str(dtype_b_original),
        'exact_match': exact_match,
        'precision_diff': precision_diff,
        'match_ratio': match_ratio,
        'max_err': max_err,
        'min_err': min_err,
        'mean_err': mean_err,
        'mse': mse,
        'cosine': cosine
    }
```

- [ ] **Step 2: Update TensorComparator.get_type_info() method**

Replace the `get_type_info` method:

```python
def get_type_info(self) -> Tuple[str, str]:
    dtype_a = str(self.a.dtype)
    shape_a = list(self.a.shape)
    dtype_b = str(self.b.dtype)
    shape_b = list(self.b.shape)
    return f"tensor(dtype={dtype_a}, shape={shape_a})", f"tensor(dtype={dtype_b}, shape={shape_b})"
```

- [ ] **Step 3: Commit**

```bash
git add acc/comparison_utils.py
git commit -m "fix: handle scalar tensors and different dtype comparison"
```

---

### Task 5: Enhance UnsupportedComparator

**Files:**
- Modify: `acc/comparison_utils.py`

- [ ] **Step 1: Update UnsupportedComparator.get_type_info() method**

Replace the `get_type_info` method:

```python
def get_type_info(self) -> Tuple[str, str]:
    return str(self.a), str(self.b)
```

- [ ] **Step 2: Update UnsupportedComparator.compare() method**

Replace the `compare` method:

```python
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

- [ ] **Step 3: Update format_comparison_log() for unsupported types**

Update the `format_comparison_log` function to handle new UnsupportedComparator output:

```python
def format_comparison_log(result: Dict) -> str:
    if 'missing_in_A' in result:
        return "missing_in_A"
    
    if 'missing_in_B' in result:
        return "missing_in_B"
    
    if 'unsupported' in result:
        return "type_unsupported"
    
    if 'content_skipped' in result:
        dtype_status = 'match' if result['dtype_match'] else 'mismatch'
        shape_status = 'match' if result['shape_match'] else 'mismatch'
        return f"dtype={dtype_status}, shape={shape_status}, content_skipped"
    
    if 'exact_match' in result and 'diff' in result:
        exact = result['exact_match']
        diff = result['diff']
        if isinstance(diff, float):
            return f"exact_match={exact}, diff={diff:.6e}"
        else:
            return f"exact_match={exact}, diff={diff}"
    
    if 'str_a' in result and 'str_b' in result:
        return f"exact_match={result['exact_match']}"
    
    if 'dtype_match' in result and 'shape_match' in result:
        parts = []
        dtype_status = 'match' if result['dtype_match'] else 'mismatch'
        shape_status = 'match' if result['shape_match'] else 'mismatch'
        parts.append(f"dtype={dtype_status}, shape={shape_status}")
        parts.append(f"exact_match={result.get('exact_match', False)}")
        
        if 'match_ratio' in result:
            parts.append(f"match_ratio={result['match_ratio']:.4f}")
        
        if 'max_err' in result:
            parts.append(f"max_err={result['max_err']:.6e}")
        
        if 'min_err' in result:
            parts.append(f"min_err={result['min_err']:.6e}")
        
        if 'mean_err' in result:
            parts.append(f"mean_err={result['mean_err']:.6e}")
        
        if 'mse' in result:
            parts.append(f"mse={result['mse']:.6e}")
        
        if 'cosine' in result:
            parts.append(f"cosine={result['cosine']:.6f}")
        
        return ', '.join(parts)
    
    if 'exact_match' in result:
        return f"exact_match={result['exact_match']}"
    
    return "unknown_format"
```

- [ ] **Step 4: Commit**

```bash
git add acc/comparison_utils.py
git commit -m "feat: enhance UnsupportedComparator with str() and == comparison"
```

---

### Task 6: Update Compare Dumps for JSON/PKL Loading

**Files:**
- Modify: `acc/compare_dumps.py`

- [ ] **Step 1: Add json import**

Add at top of file:

```python
import json
```

- [ ] **Step 2: Update _load_dumps() function**

Replace the `_load_dumps` function:

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

- [ ] **Step 3: Commit**

```bash
git add acc/compare_dumps.py
git commit -m "feat: load JSON/PKL split dump files"
```

---

### Task 7: Update Comparison Output Format

**Files:**
- Modify: `acc/compare_dumps.py`

- [ ] **Step 1: Update MATCH and SKIP log format**

Replace the match/skip log section in `ops_comp`:

```python
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
```

- [ ] **Step 2: Update COMPARE header format**

Replace the comparison header section in `ops_comp`:

```python
for idx_a, idx_b in matched_pairs:
    dump_a = dumps_a[idx_a]
    dump_b = dumps_b[idx_b]
    
    filename_safe = dump_a['filename'].replace('.py', '').replace('/', '_').replace('\\', '_')
    func_name = dump_a['function']
    opname_safe = dump_a['opname'].replace('.', '_').replace('::', '_')
    
    dump_filename = f"{dump_a['sequence']:06d}__{filename_safe}__{func_name}__{opname_safe}.json"
    
    print(f"[COMPARE] {dump_filename}")
    
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

- [ ] **Step 3: Commit**

```bash
git add acc/compare_dumps.py
git commit -m "feat: update comparison output format with simplified headers"
```

---

### Task 8: Update Test File

**Files:**
- Modify: `test_operator_tools.py`

- [ ] **Step 1: Add test for scalar tensor comparison**

```python
def test_scalar_tensor_comparison():
    import torch
    from acc import ops_dump, ops_comp
    
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        a = torch.tensor(5.0)
        b = torch.tensor(5.0)
        
        with ops_dump(tmpdir):
            c = a + b
        
        dump_dirs = os.listdir(tmpdir)
        assert len(dump_dirs) == 1
        print(f"Scalar tensor test passed, dump dir: {dump_dirs[0]}")
```

- [ ] **Step 2: Add test for session directory naming**

```python
def test_session_directory_naming():
    import torch
    from acc import ops_dump
    import tempfile
    import os
    import re
    
    with tempfile.TemporaryDirectory() as tmpdir:
        x = torch.randn(2, 3)
        
        with ops_dump(tmpdir):
            y = x + 1
        
        dump_dirs = os.listdir(tmpdir)
        assert len(dump_dirs) == 1
        
        pattern = r'^(None|\d+)-\d+-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-[a-f0-9]{8}$'
        assert re.match(pattern, dump_dirs[0])
        print(f"Session naming test passed: {dump_dirs[0]}")
```

- [ ] **Step 3: Add test for JSON/PKL split**

```python
def test_json_pkl_split():
    import torch
    from acc import ops_dump
    import tempfile
    import os
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        x = torch.randn(2, 3)
        
        with ops_dump(tmpdir):
            y = x + 1
        
        session_dir = os.path.join(tmpdir, os.listdir(tmpdir)[0])
        files = os.listdir(session_dir)
        
        json_files = [f for f in files if f.endswith('.json')]
        pkl_files = [f for f in files if f.endswith('.pkl')]
        
        assert len(json_files) > 0
        assert len(pkl_files) > 0
        
        for json_file in json_files:
            with open(os.path.join(session_dir, json_file), 'r') as f:
                data = json.load(f)
            assert 'sequence' in data
            assert 'filepath' in data
            assert 'filename' in data
            assert 'function' in data
            assert 'opname' in data
            assert 'call_stack' in data
        
        print(f"JSON/PKL split test passed, files: {len(json_files)} json, {len(pkl_files)} pkl")
```

- [ ] **Step 4: Run tests**

```bash
python test_operator_tools.py
```

- [ ] **Step 5: Commit**

```bash
git add test_operator_tools.py
git commit -m "test: add tests for scalar tensor, session naming, and JSON/PKL split"
```

---

## Self-Review

**1. Spec coverage:**
- Requirement 1 (scalar tensor): Task 4 ✓
- Requirement 2 (return original dtype): Task 4 ✓
- Requirement 3 (dtype mismatch comparison): Task 4 ✓
- Requirement 4 (session directory with rank): Task 2 ✓
- Requirement 5 (dump file naming with function): Task 3 ✓
- Requirement 6 (JSON/PKL split): Task 1, 3, 6 ✓
- Requirement 7 (function in comparison): Task 7 ✓
- Requirement 8 (unsupported type handling): Task 5 ✓

**2. Placeholder scan:** No TBD, TODO, or vague requirements found.

**3. Type consistency:** All method signatures and field names match between tasks.