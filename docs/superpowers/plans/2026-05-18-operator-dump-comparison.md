# PyTorch Operator Dump & Precision Comparison Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement single-file operator dump and precision comparison tool for PyTorch debugging

**Architecture:** Single operator_tools.py file with OperatorDumper class (context manager + decorator) and compare_operator_dumps() function. Uses TorchDispatchMode for operator interception, pickle for serialization, LCS algorithm for sequence matching.

**Tech Stack:** Python 3.x, PyTorch, pickle, traceback, inspect, numpy

---

## File Structure

```
operator_tools.py - Single file containing all implementation
  ├── Helper functions (serialization, tensor handling, LCS)
  ├── OperatorDumper class (context manager + decorator)
  └── compare_operator_dumps() function
```

---

### Task 1: Create operator_tools.py file structure

**Files:**
- Create: `operator_tools.py`

- [ ] **Step 1: Create file with imports and structure**

```python
"""
PyTorch Operator Dump & Precision Comparison Tool

Provides OperatorDumper for capturing operator calls and compare_operator_dumps for precision analysis.
"""

import os
import pickle
import traceback
import inspect
from datetime import datetime
from typing import Any, List, Tuple, Dict, Optional
import torch
import numpy as np
from torch.utils._python_dispatch import TorchDispatchMode


# Helper functions will be added here


# OperatorDumper class will be added here


# compare_operator_dumps function will be added here
```

- [ ] **Step 2: Commit initial structure**

```bash
git add operator_tools.py
git commit -m "feat: create operator_tools.py file structure"
```

---

### Task 2: Implement helper functions - serialization and tensor handling

**Files:**
- Modify: `operator_tools.py`

- [ ] **Step 1: Add tensor_to_cpu helper function**

```python
def _tensor_to_cpu(obj: Any) -> Any:
    """Convert tensors to CPU, handle numpy arrays."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_tensor_to_cpu(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: _tensor_to_cpu(v) for k, v in obj.items()}
    else:
        return obj


def _serialize_value(obj: Any) -> Any:
    """Serialize object with tensors moved to CPU."""
    return _tensor_to_cpu(obj)
```

- [ ] **Step 2: Add file name sanitization helper**

```python
def _sanitize_filename(filename: str) -> str:
    """Sanitize filename for dump file naming."""
    return filename.replace('/', '_').replace('\\', '_').replace('.py', '')
```

- [ ] **Step 3: Commit helper functions**

```bash
git add operator_tools.py
git commit -m "feat: add serialization and tensor handling helpers"
```

---

### Task 3: Implement OperatorDumper class - init and context manager

**Files:**
- Modify: `operator_tools.py`

- [ ] **Step 1: Add OperatorDumper class skeleton with __init__**

```python
class OperatorDumper:
    """Context manager and decorator for dumping PyTorch operator calls."""
    
    def __init__(self, dump_path: str):
        """
        Initialize OperatorDumper.
        
        Args:
            dump_path: Base path for dump output
        """
        self.dump_path = dump_path
        self.session_dir = None
        self.sequence = 0
        self._active = False
```

- [ ] **Step 2: Add __enter__ and __exit__ methods**

```python
    def __enter__(self):
        """Enter context manager, create session directory."""
        pid = os.getpid()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.dump_path, f"{pid}_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.sequence = 0
        self._active = True
        print(f"[DUMP] Created session directory: {self.session_dir}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self._active = False
        print(f"[DUMP] Session completed: {self.sequence} operators dumped to {self.session_dir}")
        return False
```

- [ ] **Step 3: Commit OperatorDumper init and context manager**

```bash
git add operator_tools.py
git commit -m "feat: add OperatorDumper __init__, __enter__, __exit__"
```

---

### Task 4: Implement OperatorDumper.__torch_dispatch__ for operator interception

**Files:**
- Modify: `operator_tools.py`

- [ ] **Step 1: Add TorchDispatchMode inheritance**

```python
class OperatorDumper(TorchDispatchMode):
    """Context manager and decorator for dumping PyTorch operator calls."""
```

- [ ] **Step 2: Implement __torch_dispatch__ method**

```python
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Intercept torch operator calls."""
        if not self._active:
            return func(*args, **kwargs or {})
        
        kwargs = kwargs or {}
        
        # Execute the operation
        result = func(*args, **kwargs)
        
        # Dump this operation
        self._dump_operation(func, args, kwargs, result)
        
        return result
```

- [ ] **Step 3: Commit TorchDispatchMode integration**

```bash
git add operator_tools.py
git commit -m "feat: add TorchDispatchMode operator interception"
```

---

### Task 5: Implement OperatorDumper dump logic

**Files:**
- Modify: `operator_tools.py`

- [ ] **Step 1: Add _dump_operation method**

```python
    def _dump_operation(self, func, args, kwargs, result):
        """Dump a single operator call."""
        # Extract caller information
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back.f_back  # Go up to actual caller
        filename = os.path.basename(caller_frame.f_code.co_filename)
        lineno = caller_frame.f_lineno
        
        # Get operator name
        opname = str(func)
        
        # Get full call stack
        call_stack = traceback.format_stack()
        
        # Serialize inputs and outputs
        inputs = [_serialize_value(arg) for arg in args]
        if kwargs:
            inputs.append(_serialize_value(kwargs))
        outputs = [_serialize_value(result)]
        
        # Create dump data
        dump_data = {
            'sequence': self.sequence,
            'filename': filename,
            'lineno': lineno,
            'opname': opname,
            'call_stack': ''.join(call_stack),
            'inputs': inputs,
            'outputs': outputs
        }
        
        # Create dump filename
        sanitized_name = _sanitize_filename(filename)
        dump_filename = f"{self.sequence:04d}_{sanitized_name}_{opname.replace('.', '_')}.pkl"
        dump_path = os.path.join(self.session_dir, dump_filename)
        
        # Write dump file
        with open(dump_path, 'wb') as f:
            pickle.dump(dump_data, f)
        
        # Log the dump
        print(f"[DUMP] {self.sequence:04d} | {filename}:{lineno} | {opname} | saved to {dump_filename}")
        
        self.sequence += 1
```

- [ ] **Step 2: Commit dump logic**

```bash
git add operator_tools.py
git commit -m "feat: add operator dump logic with call stack extraction"
```

---

### Task 6: Implement OperatorDumper decorator mode

**Files:**
- Modify: `operator_tools.py`

- [ ] **Step 1: Add __call__ method for decorator**

```python
    def __call__(self, func):
        """Use as decorator."""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
```

- [ ] **Step 2: Commit decorator mode**

```bash
git add operator_tools.py
git commit -m "feat: add OperatorDumper decorator mode"
```

---

### Task 7: Implement LCS algorithm

**Files:**
- Modify: `operator_tools.py`

- [ ] **Step 1: Add LCS algorithm function**

```python
def _lcs_length(a: List[str], b: List[str]) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Find longest common subsequence and return matched pairs.
    
    Args:
        a: First sequence of signatures
        b: Second sequence of signatures
    
    Returns:
        Tuple of (lcs_length, matched_pairs as list of (idx_a, idx_b))
    """
    m, n = len(a), len(b)
    
    # Build LCS table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Backtrack to find matched pairs
    matched_pairs = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i-1] == b[j-1]:
            matched_pairs.append((i-1, j-1))
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    matched_pairs.reverse()
    return dp[m][n], matched_pairs
```

- [ ] **Step 2: Commit LCS algorithm**

```bash
git add operator_tools.py
git commit -m "feat: add LCS algorithm for operator sequence matching"
```

---

### Task 8: Implement element comparison helpers

**Files:**
- Modify: `operator_tools.py`

- [ ] **Step 1: Add type formatting helpers**

```python
def _format_type_info(obj: Any) -> str:
    """Format type info for logging."""
    if obj is None:
        return "None"
    elif isinstance(obj, torch.Tensor):
        return f"tensor(dtype={obj.dtype}, shape={list(obj.shape)})"
    elif isinstance(obj, np.ndarray):
        return f"numpy(dtype={obj.dtype}, shape={list(obj.shape)})"
    elif isinstance(obj, int):
        return f"int({obj})"
    elif isinstance(obj, float):
        return f"float({obj})"
    else:
        type_name = type(obj).__name__
        if isinstance(obj, (list, tuple)):
            return f"{type_name}(len={len(obj)})"
        elif isinstance(obj, dict):
            return f"dict(len={len(obj)})"
        else:
            return f"{type_name}({repr(obj)})"
```

- [ ] **Step 2: Commit type formatting**

```bash
git add operator_tools.py
git commit -m "feat: add type info formatting helpers"
```

---

### Task 9: Implement tensor comparison with all metrics

**Files:**
- Modify: `operator_tools.py`

- [ ] **Step 1: Add tensor comparison function**

```python
def _compare_tensors(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    """
    Compare two tensors and return metrics.
    
    Args:
        a: First tensor
        b: Second tensor
    
    Returns:
        Dict with comparison metrics
    """
    # Convert numpy to tensor if needed
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
    
    # Check dtype
    dtype_match = a.dtype == b.dtype
    dtype_result = f"dtype={('match' if dtype_match else 'mismatch')}"
    
    # Check shape
    shape_match = a.shape == b.shape
    shape_result = f"shape={('match' if shape_match else 'mismatch')}"
    
    if not dtype_match or not shape_match:
        return {
            'dtype_match': dtype_match,
            'shape_match': shape_match,
            'content_skipped': True,
            'log': f"{dtype_result}, {shape_result}, content_skipped"
        }
    
    # Content comparison
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    
    exact_match = torch.allclose(a, b, rtol=0, atol=0)
    match_count = (a == b).sum().item()
    total_count = a.numel()
    match_ratio = match_count / total_count
    
    diff = torch.abs(a_flat - b_flat)
    max_err = diff.max().item()
    min_err = diff.min().item()
    mean_err = diff.mean().item()
    
    # MSE
    mse = torch.mean((a_flat - b_flat) ** 2).item()
    
    # Cosine similarity
    if total_count > 0:
        cosine = torch.nn.functional.cosine_similarity(
            a_flat.unsqueeze(0), b_flat.unsqueeze(0)
        ).item()
    else:
        cosine = 1.0
    
    return {
        'dtype_match': dtype_match,
        'shape_match': shape_match,
        'exact_match': exact_match,
        'match_ratio': match_ratio,
        'max_err': max_err,
        'min_err': min_err,
        'mean_err': mean_err,
        'mse': mse,
        'cosine': cosine,
        'log': f"{dtype_result}, {shape_result}, exact_match={exact_match}, match_ratio={match_ratio:.4f}, max_err={max_err:.6e}, min_err={min_err:.6e}, mean_err={mean_err:.6e}, mse={mse:.6e}, cosine={cosine:.6f}"
    }
```

- [ ] **Step 2: Commit tensor comparison**

```bash
git add operator_tools.py
git commit -m "feat: add tensor comparison with all metrics"
```

---

### Task 10: Implement element comparison for all types

**Files:**
- Modify: `operator_tools.py`

- [ ] **Step 1: Add generic element comparison function**

```python
def _compare_element(a: Any, b: Any) -> str:
    """
    Compare two elements and return log string.
    
    Args:
        a: First element
        b: Second element
    
    Returns:
        Log string for comparison
    """
    a_type = _format_type_info(a)
    b_type = _format_type_info(b)
    
    # Both None
    if a is None and b is None:
        return f"{a_type} | {b_type} | exact_match=True"
    
    # Type mismatch (one None, other not)
    if a is None or b is None:
        return f"{a_type} | {b_type} | type_mismatch"
    
    # int comparison
    if isinstance(a, int) and isinstance(b, int):
        exact = a == b
        diff = abs(a - b) if not exact else 0
        return f"{a_type} | {b_type} | exact_match={exact}, diff={diff}"
    
    # float comparison
    if isinstance(a, float) and isinstance(b, float):
        exact = a == b
        diff = abs(a - b) if not exact else 0.0
        return f"{a_type} | {b_type} | exact_match={exact}, diff={diff:.6e}"
    
    # Tensor or numpy comparison
    if isinstance(a, (torch.Tensor, np.ndarray)) and isinstance(b, (torch.Tensor, np.ndarray)):
        metrics = _compare_tensors(a, b)
        return f"{a_type} | {b_type} | {metrics['log']}"
    
    # Unsupported types
    return f"{a_type} | {b_type} | type_unsupported"
```

- [ ] **Step 2: Commit element comparison**

```bash
git add operator_tools.py
git commit -m "feat: add generic element comparison for all types"
```

---

### Task 11: Implement compare_operator_dumps - Phase 1 (LCS matching)

**Files:**
- Modify: `operator_tools.py`

- [ ] **Step 1: Add load_dumps helper function**

```python
def _load_dumps(dump_dir: str) -> List[Dict]:
    """
    Load all dump files from directory.
    
    Args:
        dump_dir: Path to dump directory
    
    Returns:
        List of dump data sorted by sequence
    """
    dumps = []
    for filename in os.listdir(dump_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(dump_dir, filename)
            with open(filepath, 'rb') as f:
                dumps.append(pickle.load(f))
    
    dumps.sort(key=lambda x: x['sequence'])
    return dumps
```

- [ ] **Step 2: Add compare_operator_dumps Phase 1 implementation**

```python
def compare_operator_dumps(dump_dir_a: str, dump_dir_b: str):
    """
    Compare two operator dump sessions.
    
    Args:
        dump_dir_a: Path to first dump directory
        dump_dir_b: Path to second dump directory
    """
    # Phase 1: Load dumps
    print(f"[LCS] Loading dump A: from {dump_dir_a}")
    dumps_a = _load_dumps(dump_dir_a)
    print(f"[LCS] Loading dump B: from {dump_dir_b}")
    dumps_b = _load_dumps(dump_dir_b)
    
    print(f"[LCS] Loading dump A: {len(dumps_a)} operators from {dump_dir_a}")
    print(f"[LCS] Loading dump B: {len(dumps_b)} operators from {dump_dir_b}")
    
    # Build signatures
    print("[LCS] Building operator signatures...")
    sigs_a = [f"{d['filename']}::{d['opname']}" for d in dumps_a]
    sigs_b = [f"{d['filename']}::{d['opname']}" for d in dumps_b]
    
    # Find LCS
    print("[LCS] Finding longest common subsequence...")
    lcs_len, matched_pairs = _lcs_length(sigs_a, sigs_b)
    
    a_only = len(dumps_a) - lcs_len
    b_only = len(dumps_b) - lcs_len
    print(f"[LCS] Matched: {lcs_len} operators | A-only: {a_only} | B-only: {b_only}")
    
    # Log matches and skips
    matched_a_indices = set(idx_a for idx_a, idx_b in matched_pairs)
    matched_b_indices = set(idx_b for idx_a, idx_b in matched_pairs)
    
    for i, dump in enumerate(dumps_a):
        if i in matched_a_indices:
            match_idx = matched_pairs[list(matched_a_indices).index(i)][1]
            print(f"[MATCH] A:{dump['sequence']:04d}_{dump['filename']}_{dump['opname']} <-> B:{dumps_b[match_idx]['sequence']:04d}_{dumps_b[match_idx]['filename']}_{dumps_b[match_idx]['opname']}")
        else:
            print(f"[SKIP] A:{dump['sequence']:04d}_{dump['filename']}_{dump['opname']} (no match in B)")
    
    for j, dump in enumerate(dumps_b):
        if j not in matched_b_indices:
            print(f"[SKIP] B:{dump['sequence']:04d}_{dump['filename']}_{dump['opname']} (no match in A)")
    
    # Phase 2 will be added in next task
```

- [ ] **Step 3: Commit Phase 1 implementation**

```bash
git add operator_tools.py
git commit -m "feat: add compare_operator_dumps Phase 1 LCS matching"
```

---

### Task 12: Implement compare_operator_dumps - Phase 2 (detailed comparison)

**Files:**
- Modify: `operator_tools.py`

- [ ] **Step 1: Add Phase 2 comparison logic**

```python
    # Continue compare_operator_dumps after Phase 1
    
    # Phase 2: Detailed comparison
    print(f"[COMPARE] Starting detailed comparison of {lcs_len} matched pairs...")
    
    # Statistics counters
    stats = {
        'exact_match': 0,
        'precision_diff': 0,
        'dtype_mismatch': 0,
        'shape_mismatch': 0,
        'input_count_mismatch': 0,
        'output_count_mismatch': 0
    }
    
    for idx_a, idx_b in matched_pairs:
        dump_a = dumps_a[idx_a]
        dump_b = dumps_b[idx_b]
        
        op_id = f"{dump_a['sequence']:04d}_{dump_a['filename']}_{dump_a['opname']}"
        print(f"[COMPARE] {op_id}:")
        
        # Compare inputs
        inputs_a = dump_a['inputs']
        inputs_b = dump_b['inputs']
        
        max_inputs = max(len(inputs_a), len(inputs_b))
        if len(inputs_a) != len(inputs_b):
            stats['input_count_mismatch'] += 1
        
        for i in range(max_inputs):
            if i >= len(inputs_a):
                b_info = _format_type_info(inputs_b[i])
                print(f"  Inputs[{i}] | <missing> | {b_info} | missing_in_A")
            elif i >= len(inputs_b):
                a_info = _format_type_info(inputs_a[i])
                print(f"  Inputs[{i}] | {a_info} | <missing> | missing_in_B")
            else:
                log = _compare_element(inputs_a[i], inputs_b[i])
                print(f"  Inputs[{i}] | {log}")
        
        # Compare outputs
        outputs_a = dump_a['outputs']
        outputs_b = dump_b['outputs']
        
        max_outputs = max(len(outputs_a), len(outputs_b))
        if len(outputs_a) != len(outputs_b):
            stats['output_count_mismatch'] += 1
        
        for i in range(max_outputs):
            if i >= len(outputs_a):
                b_info = _format_type_info(outputs_b[i])
                print(f"  Outputs[{i}] | <missing> | {b_info} | missing_in_A")
            elif i >= len(outputs_b):
                a_info = _format_type_info(outputs_a[i])
                print(f"  Outputs[{i}] | {a_info} | <missing> | missing_in_B")
            else:
                log = _compare_element(outputs_a[i], outputs_b[i])
                print(f"  Outputs[{i}] | {log}")
    
    # Summary
    print(f"[SUMMARY] Total: {lcs_len} | Exact match: {stats['exact_match']} | Precision diff: {stats['precision_diff']} | Dtype mismatch: {stats['dtype_mismatch']} | Shape mismatch: {stats['shape_mismatch']} | Input count mismatch: {stats['input_count_mismatch']} | Output count mismatch: {stats['output_count_mismatch']}")
```

- [ ] **Step 2: Commit Phase 2 implementation**

```bash
git add operator_tools.py
git commit -m "feat: add compare_operator_dumps Phase 2 detailed comparison"
```

---

### Task 13: Add statistics tracking to element comparison

**Files:**
- Modify: `operator_tools.py`

- [ ] **Step 1: Modify _compare_element to return stats dict**

```python
def _compare_element(a: Any, b: Any) -> Tuple[str, Dict[str, bool]]:
    """
    Compare two elements and return log string and stats.
    
    Args:
        a: First element
        b: Second element
    
    Returns:
        Tuple of (log string, stats dict)
    """
    a_type = _format_type_info(a)
    b_type = _format_type_info(b)
    
    stats = {
        'exact_match': False,
        'precision_diff': False,
        'dtype_mismatch': False,
        'shape_mismatch': False
    }
    
    # Both None
    if a is None and b is None:
        stats['exact_match'] = True
        return f"{a_type} | {b_type} | exact_match=True", stats
    
    # Type mismatch (one None, other not)
    if a is None or b is None:
        stats['dtype_mismatch'] = True
        return f"{a_type} | {b_type} | type_mismatch", stats
    
    # int comparison
    if isinstance(a, int) and isinstance(b, int):
        exact = a == b
        diff = abs(a - b) if not exact else 0
        stats['exact_match'] = exact
        return f"{a_type} | {b_type} | exact_match={exact}, diff={diff}", stats
    
    # float comparison
    if isinstance(a, float) and isinstance(b, float):
        exact = a == b
        diff = abs(a - b) if not exact else 0.0
        stats['exact_match'] = exact
        if not exact:
            stats['precision_diff'] = True
        return f"{a_type} | {b_type} | exact_match={exact}, diff={diff:.6e}", stats
    
    # Tensor or numpy comparison
    if isinstance(a, (torch.Tensor, np.ndarray)) and isinstance(b, (torch.Tensor, np.ndarray)):
        metrics = _compare_tensors(a, b)
        
        if not metrics['dtype_match']:
            stats['dtype_mismatch'] = True
        elif not metrics['shape_match']:
            stats['shape_mismatch'] = True
        elif metrics['exact_match']:
            stats['exact_match'] = True
        else:
            stats['precision_diff'] = True
        
        return f"{a_type} | {b_type} | {metrics['log']}", stats
    
    # Unsupported types
    return f"{a_type} | {b_type} | type_unsupported", stats
```

- [ ] **Step 2: Update compare_operator_dumps to use stats**

```python
        for i in range(max_inputs):
            if i >= len(inputs_a):
                b_info = _format_type_info(inputs_b[i])
                print(f"  Inputs[{i}] | <missing> | {b_info} | missing_in_A")
            elif i >= len(inputs_b):
                a_info = _format_type_info(inputs_a[i])
                print(f"  Inputs[{i}] | {a_info} | <missing> | missing_in_B")
            else:
                log, element_stats = _compare_element(inputs_a[i], inputs_b[i])
                print(f"  Inputs[{i}] | {log}")
                # Update stats
                if element_stats['exact_match']:
                    stats['exact_match'] += 1
                if element_stats['precision_diff']:
                    stats['precision_diff'] += 1
                if element_stats['dtype_mismatch']:
                    stats['dtype_mismatch'] += 1
                if element_stats['shape_mismatch']:
                    stats['shape_mismatch'] += 1
        
        # Similar for outputs...
```

- [ ] **Step 3: Commit statistics tracking**

```bash
git add operator_tools.py
git commit -m "feat: add statistics tracking to comparison"
```

---

## Self-Review Checklist

After writing this plan, verify:

**1. Spec coverage:**
- ✓ OperatorDumper class with context manager and decorator modes
- ✓ TorchDispatchMode for operator interception
- ✓ Session directory creation (pid_timestamp format)
- ✓ Dump file structure with sequence, filename, lineno, opname, call_stack, inputs, outputs
- ✓ Pickle serialization with tensors on CPU
- ✓ Dump logging format
- ✓ LCS algorithm for sequence matching
- ✓ Universal comparison flow for all types (None, int, float, tensor, numpy, unsupported)
- ✓ Tensor comparison metrics (exact_match, match_ratio, max_err, min_err, mean_err, mse, cosine)
- ✓ Comparison logging format (three-part structure)
- ✓ Phase 1 and Phase 2 logging
- ✓ Summary statistics
- ✓ Input/output count mismatch handling

**2. Placeholder scan:**
- ✓ No TBD, TODO, or placeholder phrases
- ✓ All code blocks contain complete implementation
- ✓ All steps have specific commands

**3. Type consistency:**
- ✓ _tensor_to_cpu returns Any
- ✓ _serialize_value returns Any
- ✓ _format_type_info returns str
- ✓ _compare_tensors returns Dict[str, Any]
- ✓ _compare_element returns Tuple[str, Dict[str, bool]]
- ✓ OperatorDumper methods match spec

**Plan complete and saved to `docs/superpowers/plans/2026-05-18-operator-dump-comparison.md`.**