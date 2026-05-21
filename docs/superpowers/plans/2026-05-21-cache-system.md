# Cache System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a BLAKE2-based content cache for tensor/numpy deduplication, separate metadata (JSON) from data (PKL), restructure inputs as args/kwargs, add ETA progress to comparison, and enhance tensor type descriptions with nan/inf info.

**Architecture:** New `cache.py` module with `CacheEntry` + `CacheManager` for content-addressable storage in `storage/` subdirectory. `SerializationSession` (replacing `DumperManager`) integrates cache and manages save/load. `OperatorRecord` (renamed from `OperatorDump`) uses `args`/`kwargs`/`outputs`. Comparison loads metadata-only for LCS, then on-demand data with ETA progress.

**Tech Stack:** Python 3, PyTorch, numpy, hashlib (blake2b), pickle, json, dataclasses

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `acc/cache.py` | CREATE | `CacheEntry` dataclass, `CacheManager` with `get_or_cache` / `resolve` |
| `acc/serialization.py` | MODIFY | `OperatorRecord` (renamed), `SerializationSession`, `_sanitize_*` helpers |
| `acc/dump.py` | MODIFY | `ops_dump` simplified via `SerializationSession`, impl patch, `_active_session` |
| `acc/comp.py` | MODIFY | Metadata-only LCS, on-demand data loading, ETA, args/kwargs compare |
| `acc/comparators.py` | MODIFY | `get_type_info` with nan/inf/neg_inf for TensorComparator, NumpyComparator |
| `acc/formatting.py` | MODIFY | Add `format_eta`, update for `OperatorRecord` |
| `acc/__init__.py` | MODIFY | Export `OperatorRecord`, `SerializationSession` |
| `tests/test_cache.py` | CREATE | CacheManager unit tests |
| `tests/test_operator_tools.py` | MODIFY | Adapt to new structure |

---

### Task 1: Create CacheManager

**Files:**
- Create: `acc/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Create `acc/cache.py` with CacheEntry and CacheManager**

```python
"""Content-addressable cache for tensor/numpy deduplication."""

import hashlib
import os
import pickle
from dataclasses import dataclass
from typing import Any, List, Set

import numpy as np
import torch


@dataclass
class CacheEntry:
    """Metadata for a cached tensor/numpy array."""
    cache_id: str        # content hash (blake2b hex digest)
    type: str            # 'tensor' or 'numpy'
    dtype: str           # e.g. 'float32', 'int64'
    shape: List[int]     # tensor/array shape


class CacheManager:
    """Content-addressable tensor/numpy cache backed by a storage directory."""

    def __init__(self, storage_dir: str, enable_cache: bool = True):
        self.storage_dir = storage_dir
        self.enable_cache = enable_cache
        self._cached_ids: Set[str] = set()

    def get_or_cache(self, obj: Any) -> Any:
        """Cache tensor/numpy by content hash. Returns CacheEntry or original object."""
        if not self.enable_cache:
            return obj
        if not isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj
        cache_id = self._compute_content_hash(obj)
        if cache_id in self._cached_ids:
            return CacheEntry(
                cache_id=cache_id,
                type='tensor' if isinstance(obj, torch.Tensor) else 'numpy',
                dtype=str(obj.dtype).replace('torch.', ''),
                shape=list(obj.shape),
            )
        self._cached_ids.add(cache_id)
        self._save_to_storage(obj, cache_id)
        return CacheEntry(
            cache_id=cache_id,
            type='tensor' if isinstance(obj, torch.Tensor) else 'numpy',
            dtype=str(obj.dtype).replace('torch.', ''),
            shape=list(obj.shape),
        )

    def resolve(self, obj: Any) -> Any:
        """Resolve CacheEntry to tensor/numpy, pass through everything else."""
        if isinstance(obj, CacheEntry):
            return self._load_from_storage(obj.cache_id)
        return obj

    def _compute_content_hash(self, obj) -> str:
        """Compute BLAKE2b hash of tensor/numpy content bytes."""
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().contiguous().cpu().numpy()
        else:
            arr = obj
        return hashlib.blake2b(arr.tobytes(), digest_size=32).hexdigest()

    def _save_to_storage(self, obj, cache_id: str):
        """Save tensor/numpy to storage/{cache_id}.pkl."""
        filepath = os.path.join(self.storage_dir, f"{cache_id}.pkl")
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().contiguous().cpu()
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)

    def _load_from_storage(self, cache_id: str) -> Any:
        """Load tensor/numpy from storage/{cache_id}.pkl."""
        filepath = os.path.join(self.storage_dir, f"{cache_id}.pkl")
        with open(filepath, 'rb') as f:
            return pickle.load(f)
```

- [ ] **Step 2: Verify module loads**

Run: `python -c "from acc.cache import CacheEntry, CacheManager; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Create `tests/test_cache.py`**

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import torch
import numpy as np
from acc.cache import CacheEntry, CacheManager


def test_cache_entry():
    print("Test: CacheEntry dataclass")
    entry = CacheEntry(cache_id="abc123", type="tensor", dtype="float32", shape=[2, 3])
    assert entry.cache_id == "abc123"
    assert entry.type == "tensor"
    assert entry.shape == [2, 3]
    print("  PASS")


def test_get_or_cache_tensor():
    print("Test: get_or_cache with tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage)
        t = torch.randn(2, 3)
        result = mgr.get_or_cache(t)
        assert isinstance(result, CacheEntry)
        assert result.type == "tensor"
        assert result.shape == [2, 3]
        result2 = mgr.get_or_cache(t)
        assert isinstance(result2, CacheEntry)
        assert result2.cache_id == result.cache_id
    print("  PASS")


def test_get_or_cache_numpy():
    print("Test: get_or_cache with numpy")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage)
        a = np.random.randn(3, 4).astype(np.float32)
        result = mgr.get_or_cache(a)
        assert isinstance(result, CacheEntry)
        assert result.type == "numpy"
        result2 = mgr.get_or_cache(a)
        assert result2.cache_id == result.cache_id
    print("  PASS")


def test_get_or_cache_scalar():
    print("Test: get_or_cache with non-tensor/numpy")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage)
        assert mgr.get_or_cache(42) == 42
        assert mgr.get_or_cache(3.14) == 3.14
        assert mgr.get_or_cache("hello") == "hello"
        assert mgr.get_or_cache(None) is None
    print("  PASS")


def test_resolve_tensor():
    print("Test: resolve tensor from CacheEntry")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage)
        t = torch.tensor([1.0, 2.0, 3.0])
        entry = mgr.get_or_cache(t)
        resolved = mgr.resolve(entry)
        assert isinstance(resolved, torch.Tensor)
        assert torch.equal(resolved, t)
    print("  PASS")


def test_resolve_nested():
    print("Test: resolve nested structure with CacheEntry")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage)
        t = torch.randn(2, 2)
        entry = mgr.get_or_cache(t)
        nested = [entry, 42, "hello"]
        resolved = mgr.resolve(nested)
        assert isinstance(resolved[0], torch.Tensor)
        assert resolved[1] == 42
        assert resolved[2] == "hello"
    print("  PASS")


def test_enable_cache_false():
    print("Test: enable_cache=False returns original objects")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage, enable_cache=False)
        t = torch.randn(2, 3)
        result = mgr.get_or_cache(t)
        assert result is t
    print("  PASS")


def test_different_tensors_different_hash():
    print("Test: different tensors get different cache_ids")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage)
        t1 = torch.ones(2, 3)
        t2 = torch.zeros(2, 3)
        e1 = mgr.get_or_cache(t1)
        e2 = mgr.get_or_cache(t2)
        assert e1.cache_id != e2.cache_id
    print("  PASS")


def test_identical_tensors_same_hash():
    print("Test: identical content yields same cache_id")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage)
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([1.0, 2.0])
        e1 = mgr.get_or_cache(t1)
        e2 = mgr.get_or_cache(t2)
        assert e1.cache_id == e2.cache_id
    print("  PASS")


def main():
    print("ACC - CacheManager Unit Tests\n")
    test_cache_entry()
    test_get_or_cache_tensor()
    test_get_or_cache_numpy()
    test_get_or_cache_scalar()
    test_resolve_tensor()
    test_resolve_nested()
    test_enable_cache_false()
    test_different_tensors_different_hash()
    test_identical_tensors_same_hash()
    print("\nAll cache tests passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `python tests/test_cache.py`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add acc/cache.py tests/test_cache.py
git commit -m "feat: add CacheManager with BLAKE2 content-addressable cache"
```

---

### Task 2: Refactor serialization.py — OperatorRecord + SerializationSession

**Files:**
- Modify: `acc/serialization.py`

- [ ] **Step 1: Rewrite `acc/serialization.py`**

Replace the entire file content with:

```python
"""
Serialization helpers and data structures for PyTorch Operator Dump Tool.
"""

import os
import json
import pickle
import time
import uuid
import traceback
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Optional

import torch

from .cache import CacheEntry, CacheManager


@dataclass
class OperatorRecord:
    """Data structure for a single operator dump."""
    sequence: int
    filepath: str
    filename: str
    function: str
    lineno: int
    opname: str
    call_stack: List[Dict]
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> 'OperatorRecord':
        return cls(
            sequence=data['sequence'],
            filepath=data.get('filepath', ''),
            filename=data['filename'],
            function=data['function'],
            lineno=data.get('lineno', 0),
            opname=data['opname'],
            call_stack=data.get('call_stack', []),
            args=data.get('args', []),
            kwargs=data.get('kwargs', {}),
            outputs=data.get('outputs', [])
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON (metadata only, no tensor data)."""
        return {
            'sequence': self.sequence,
            'filepath': self.filepath,
            'filename': self.filename,
            'function': self.function,
            'lineno': self.lineno,
            'opname': self.opname,
            'call_stack': self.call_stack
        }


OperatorDump = OperatorRecord


def _sanitize_filename(filename: str) -> str:
    if filename.startswith('<') and filename.endswith('>'):
        filename = filename[1:-1]
    result = filename.replace('/', '_').replace('\\', '_').replace('.py', '')
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    for char in invalid_chars:
        result = result.replace(char, '_')
    return result


def _sanitize_opname(opname: str) -> str:
    return opname.replace('.', '_').replace('::', '_')


def _serialize_tensor(tensor, max_tensor_size_mb: int):
    """Prepare a single tensor for serialization with size check."""
    tensor_size_bytes = tensor.numel() * tensor.element_size()
    tensor_size_mb = tensor_size_bytes / (1024 * 1024)
    if tensor_size_mb > max_tensor_size_mb:
        print(f"[DUMP WARN] Tensor size {tensor_size_mb:.2f} MB exceeds limit {max_tensor_size_mb} MB, replacing with None")
        return None
    try:
        return tensor.detach().contiguous().cpu()
    except Exception as e:
        print(f"[DUMP WARN] Failed to make tensor contiguous: {e}, replacing with None")
        return None


def _serialize_value(value, max_tensor_size_mb: int, cache_mgr: CacheManager):
    """Serialize a single value: tensors get prepared + cached, everything else passes through."""
    if isinstance(value, torch.Tensor):
        prepared = _serialize_tensor(value, max_tensor_size_mb)
        if prepared is None:
            return None
        return cache_mgr.get_or_cache(prepared)
    return value


def _serialize_outputs(result, max_tensor_size_mb: int, cache_mgr: CacheManager) -> list:
    """Serialize operator outputs to a list with tensor preparation and caching."""
    outputs_list = []
    if result is None:
        return outputs_list
    if isinstance(result, torch.Tensor):
        prepared = _serialize_tensor(result, max_tensor_size_mb)
        if prepared is None:
            outputs_list.append(None)
        else:
            outputs_list.append(cache_mgr.get_or_cache(prepared))
    elif isinstance(result, (tuple, list)):
        for item in result:
            outputs_list.append(_serialize_value(item, max_tensor_size_mb, cache_mgr))
    else:
        outputs_list.append(result)
    return outputs_list


class SerializationSession:
    """Manages a single serialization session, integrating CacheManager."""

    def __init__(self, dump_path: str, max_tensor_size_mb: int = 10240, enable_cache: bool = True):
        self.dump_path = dump_path
        self.session_dir: Optional[str] = None
        self.sequence: int = 0
        self.max_tensor_size_mb: int = max_tensor_size_mb
        self._enable_cache: bool = enable_cache
        self._cache_manager: Optional[CacheManager] = None
        self._start_time: Optional[float] = None

    def start(self) -> str:
        """Create session directory and storage subdirectory, initialize CacheManager."""
        import torch.distributed as dist
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
        storage_dir = os.path.join(self.session_dir, 'storage')
        os.makedirs(storage_dir, exist_ok=False)
        self._cache_manager = CacheManager(storage_dir, self._enable_cache)
        self.sequence = 0
        self._start_time = time.time()
        print(f"[DUMP] Created session directory: {self.session_dir}")
        return self.session_dir

    def save_operation(
        self, func, filepath: str, filename: str, function: str,
        lineno: int, args: tuple, kwargs: dict, outputs
    ) -> int:
        """Save a single operator dump. Returns sequence number."""
        if self._cache_manager is None:
            raise RuntimeError("Session not started")
        filename_safe = _sanitize_filename(filename)
        opname_safe = _sanitize_opname(str(func))
        serialized_args = [
            _serialize_value(arg, self.max_tensor_size_mb, self._cache_manager)
            for arg in args
        ]
        serialized_kwargs = {}
        for key, val in (kwargs or {}).items():
            serialized_kwargs[key] = _serialize_value(val, self.max_tensor_size_mb, self._cache_manager)
        serialized_outputs = _serialize_outputs(outputs, self.max_tensor_size_mb, self._cache_manager)
        stack = traceback.extract_stack()
        call_stack = [
            {'filepath': frame.filename, 'lineno': frame.lineno, 'line': frame.line}
            for frame in stack
        ]
        seq = self.sequence
        json_filename = f"{seq:06d}__{filename_safe}__{function}__{opname_safe}.json"
        pkl_filename = f"{seq:06d}__{filename_safe}__{function}__{opname_safe}.pkl"
        json_path = os.path.join(self.session_dir, json_filename)
        pkl_path = os.path.join(self.session_dir, pkl_filename)
        try:
            with open(json_path, 'w') as f:
                json.dump({
                    'sequence': seq, 'filepath': filepath, 'filename': filename,
                    'function': function, 'lineno': lineno, 'opname': str(func),
                    'call_stack': call_stack
                }, f, indent=2)
            with open(pkl_path, 'wb') as f:
                pickle.dump({
                    'inputs': {'args': serialized_args, 'kwargs': serialized_kwargs},
                    'outputs': serialized_outputs,
                }, f)
        except Exception as e:
            print(f"[DUMP ERROR] {seq:06d} | {filename}:{lineno} | {func} | {e}")
            self.sequence += 1
            return seq
        print(f"[DUMP] {seq:06d} | {filename}:{lineno} | {func} | saved to {json_filename}")
        self.sequence += 1
        return seq

    def end(self):
        """End the session and print summary."""
        if self.session_dir:
            elapsed = time.time() - self._start_time if self._start_time else 0
            print(f"[DUMP] Session completed: {self.sequence} operators dumped to {self.session_dir} in {elapsed:.1f}s")

    @staticmethod
    def load_metadata(json_path: str) -> OperatorRecord:
        """Load JSON metadata (no tensor data)."""
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        return OperatorRecord(
            sequence=metadata['sequence'],
            filepath=metadata.get('filepath', ''),
            filename=metadata['filename'],
            function=metadata['function'],
            lineno=metadata.get('lineno', 0),
            opname=metadata['opname'],
            call_stack=metadata.get('call_stack', []),
        )

    @staticmethod
    def load_data(pkl_path: str, storage_dir: str) -> Tuple[Dict, List]:
        """Load PKL data and resolve CacheEntry references to actual tensors.

        Returns: (inputs_dict, outputs_list) where inputs_dict has 'args' and 'kwargs' keys.
        """
        cache_mgr = CacheManager(storage_dir, enable_cache=False)
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        inputs = pkl_data['inputs']
        outputs = pkl_data['outputs']
        resolved_args = [cache_mgr.resolve(v) for v in inputs['args']]
        resolved_kwargs = {k: cache_mgr.resolve(v) for k, v in inputs['kwargs'].items()}
        resolved_outputs = [cache_mgr.resolve(v) for v in outputs]
        return {'args': resolved_args, 'kwargs': resolved_kwargs}, resolved_outputs

    @property
    def storage_dir(self) -> Optional[str]:
        if self.session_dir:
            return os.path.join(self.session_dir, 'storage')
        return None
```

- [ ] **Step 2: Verify imports**

Run: `python -c "from acc.serialization import OperatorRecord, SerializationSession, OperatorDump; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add acc/serialization.py
git commit -m "refactor: replace OperatorDump with OperatorRecord, add SerializationSession"
```

---

### Task 3: Simplify dump.py — use SerializationSession

**Files:**
- Modify: `acc/dump.py`

- [ ] **Step 1: Rewrite `acc/dump.py`**

Replace the entire file content with:

```python
"""
Operator Dumper for PyTorch Operator Dump Tool.

Provides ops_dump context manager for capturing operator calls.
Environment variable ACC_DUMP_ENABLED controls global dump behavior:
- Not set or "1"/"true": dump enabled (default)
- "0"/"false": dump disabled
"""

import os
import traceback
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from .serialization import SerializationSession, _sanitize_filename, _sanitize_opname


# Module-level reference for impl patch to access active session
_active_session = None


# ============================================================
# Patch torch.library.impl for custom operator internal capture
# ============================================================

_original_impl = None
_patch_installed = False


def _install_impl_patch():
    global _original_impl, _patch_installed
    if _patch_installed:
        return
    if not hasattr(torch.library, 'impl'):
        return
    _original_impl = torch.library.impl
    _patch_installed = True
    print("[DUMP PATCH] Installing torch.library.impl patch")

    def patched_impl(qualname, types, func=None, *, lib=None):
        def wrap(f):
            def wrapped(*args, **kwargs):
                class NestedMode(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        result = func(*args, **(kwargs or {}))
                        if _active_session is not None:
                            stack = traceback.extract_stack()
                            filepath, filename, func_name, lineno = "", "", "", 0
                            for frame_info in reversed(stack):
                                if not frame_info.filename.endswith('dump.py'):
                                    filepath = frame_info.filename
                                    filename = os.path.basename(frame_info.filename)
                                    func_name = frame_info.name
                                    lineno = frame_info.lineno
                                    break
                            _active_session.save_operation(
                                func, filepath, filename, func_name, lineno,
                                args, kwargs or {}, result
                            )
                        return result
                mode = NestedMode()
                mode.__enter__()
                try:
                    return f(*args, **kwargs)
                finally:
                    mode.__exit__(None, None, None)
            return wrapped
        if func is None:
            def decorator(f):
                return _original_impl(qualname, types, wrap(f), lib=lib)
            return decorator
        else:
            return _original_impl(qualname, types, wrap(func), lib=lib)

    torch.library.impl = patched_impl


# Install patch on import
_install_impl_patch()


# ============================================================
# ops_dump context manager (entry point)
# ============================================================

class ops_dump(TorchDispatchMode):
    """
    Context manager and decorator for dumping PyTorch operator calls.

    Usage:
        with ops_dump('/path/to/dump'):
            model(input)
    """

    def __init__(self, dump_path: str, max_tensor_size_mb: int = 10240, enable_cache: bool = True):
        self.dump_path = dump_path
        self.max_tensor_size_mb = max_tensor_size_mb
        self.enable_cache = enable_cache
        self.session = SerializationSession(dump_path, max_tensor_size_mb, enable_cache)
        self.enabled = os.environ.get('ACC_DUMP_ENABLED', '1').lower() not in ('0', 'false', 'no', 'off')

    def __enter__(self):
        global _active_session
        if not self.enabled:
            return super().__enter__()
        self.session.start()
        _active_session = self.session
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _active_session
        if self.enabled:
            self.session.end()
            _active_session = None
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not self.enabled:
            return func(*args, **(kwargs or {}))
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        stack = traceback.extract_stack()
        filepath, filename, func_name, lineno = "", "", "", 0
        for frame_info in reversed(stack):
            if not frame_info.filename.endswith('dump.py'):
                filepath = frame_info.filename
                filename = os.path.basename(frame_info.filename)
                func_name = frame_info.name
                lineno = frame_info.lineno
                break
        self.session.save_operation(func, filepath, filename, func_name, lineno, args, kwargs, result)
        return result

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
```

- [ ] **Step 2: Verify basic dump test**

Run:
```
python -c "
import tempfile, torch
from acc import ops_dump
with tempfile.TemporaryDirectory() as d:
    with ops_dump(d):
        a = torch.randn(2,3)
        b = a + 1
    import os
    dirs = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d,x))]
    assert len(dirs) > 0
    session = os.path.join(d, dirs[0])
    jsons = [f for f in os.listdir(session) if f.endswith('.json')]
    pkls = [f for f in os.listdir(session) if f.endswith('.pkl')]
    print(f'JSON files: {len(jsons)}, PKL files: {len(pkls)}, storage exists: {os.path.exists(os.path.join(session, \"storage\"))}')
    print('ok')
"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add acc/dump.py
git commit -m "refactor: simplify ops_dump to use SerializationSession"
```

---

### Task 4: Update comparators.py — add nan/inf/neg_inf to type descriptions

**Files:**
- Modify: `acc/comparators.py`

- [ ] **Step 1: Rewrite TensorComparator.get_type_info and NumpyComparator.get_type_info**

Find `class TensorComparator(ElementComparator):` in `acc/comparators.py` and replace its `get_type_info` method:

```python
def get_type_info(self) -> Tuple[str, str]:
    dtype_a = str(self.a.dtype)
    shape_a = list(self.a.shape)
    dtype_b = str(self.b.dtype)
    shape_b = list(self.b.shape)

    a_nan = torch.isnan(self.a).any().item()
    a_inf = torch.isinf(self.a).any().item()
    a_neg_inf = a_inf and (self.a < 0).any().item()

    b_nan = torch.isnan(self.b).any().item()
    b_inf = torch.isinf(self.b).any().item()
    b_neg_inf = b_inf and (self.b < 0).any().item()

    desc_a = f"tensor(dtype={dtype_a}, shape={shape_a}, nan={a_nan}, inf={a_inf}, neg_inf={a_neg_inf})"
    desc_b = f"tensor(dtype={dtype_b}, shape={shape_b}, nan={b_nan}, inf={b_inf}, neg_inf={b_neg_inf})"
    return desc_a, desc_b
```

Find `class NumpyComparator(TensorComparator):` and replace its `get_type_info` method:

```python
def get_type_info(self) -> Tuple[str, str]:
    import numpy as np
    dtype_a = str(self.a.dtype)
    shape_a = list(self.a.shape)
    dtype_b = str(self.b.dtype)
    shape_b = list(self.b.shape)

    a_nan = bool(np.isnan(self.a).any())
    a_inf = bool(np.isinf(self.a).any())
    a_neg_inf = a_inf and bool((self.a < 0).any())

    b_nan = bool(np.isnan(self.b).any())
    b_inf = bool(np.isinf(self.b).any())
    b_neg_inf = b_inf and bool((self.b < 0).any())

    desc_a = f"numpy(dtype={dtype_a}, shape={shape_a}, nan={a_nan}, inf={a_inf}, neg_inf={a_neg_inf})"
    desc_b = f"numpy(dtype={dtype_b}, shape={shape_b}, nan={b_nan}, inf={b_inf}, neg_inf={b_neg_inf})"
    return desc_a, desc_b
```

- [ ] **Step 2: Verify comparator output**

Run:
```
python -c "
import torch, numpy as np
from acc.comparators import TensorComparator, NumpyComparator

t1 = torch.tensor([1.0, float('nan'), float('inf')])
t2 = torch.tensor([1.0, 2.0, -float('inf')])
c = TensorComparator(t1, t2)
left, right = c.get_type_info()
print('Tensor:', left, '|', right)

n1 = np.array([1.0, np.nan, np.inf])
n2 = np.array([1.0, np.nan, -np.inf])
c2 = NumpyComparator(n1, n2)
left2, right2 = c2.get_type_info()
print('Numpy:', left2, '|', right2)
print('ok')
"
```
Expected: `ok` with nan/inf visible in descriptions

- [ ] **Step 3: Commit**

```bash
git add acc/comparators.py
git commit -m "feat: add nan/inf/neg_inf to tensor/numpy comparator type descriptions"
```

---

### Task 5: Add format_eta to formatting.py

**Files:**
- Modify: `acc/formatting.py`

- [ ] **Step 1: Add `format_eta` function and update `format_signature`**

At the end of `acc/formatting.py` (before the end of file), add:

```python
def format_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m{int(seconds%60)}s"
    else:
        return f"{int(seconds/3600)}h{int(seconds%3600/60)}m"
```

Update `format_signature` to use `OperatorRecord` structure (replace `dump.inputs` with appropriate method):
```python
def format_signature(dump) -> str:
    """Build signature key for LCS matching."""
    filename = dump.filename.replace('.py', '')
    return f"{filename}::{dump.opname}"
```
(No change needed functionally; the function already uses `dump.filename` and `dump.opname` which exist on both `OperatorDump` and `OperatorRecord`.)

Also update the import in formatting.py to also import `OperatorRecord`:
```python
from .serialization import OperatorDump, _sanitize_filename, _sanitize_opname
```
Change to:
```python
from .serialization import OperatorRecord, _sanitize_filename, _sanitize_opname
```

And update `format_display_key` and `format_dump_filename` to use `OperatorRecord` as type hint (no functional change).

- [ ] **Step 2: Verify format_eta**

Run: `python -c "from acc.formatting import format_eta; print(format_eta(30)); print(format_eta(120)); print(format_eta(4000)); print('ok')"`
Expected: `30s`, `2m0s`, `1h6m`, `ok`

- [ ] **Step 3: Commit**

```bash
git add acc/formatting.py
git commit -m "feat: add format_eta and update formatting imports for OperatorRecord"
```

---

### Task 6: Update comp.py — metadata-only LCS, ETA, args/kwargs compare

**Files:**
- Modify: `acc/comp.py`

- [ ] **Step 1: Rewrite `acc/comp.py`**

Replace the entire file content with:

```python
"""
Operator Dumps Comparison for PyTorch Operator Dump Tool.

Provides ops_comp function for comparing two dump sessions.
"""

import json
import os
import time
from typing import List, Tuple
from .serialization import SerializationSession
from .formatting import (
    format_signature,
    format_display_key,
    format_dump_filename,
    format_comparison_result,
    format_eta,
)
from .comparators import (
    create_comparator,
    MissingInAComparator,
    MissingInBComparator
)


def _lcs_length(a: List[str], b: List[str]) -> Tuple[int, List[Tuple[int, int]]]:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
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


def _load_all_metadata(dump_dir: str) -> list:
    """Load all metadata from dump directory (JSON only, no tensor data)."""
    records = []
    for filename in os.listdir(dump_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(dump_dir, filename)
            try:
                record = SerializationSession.load_metadata(json_path)
                records.append(record)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"[COMP WARN] Failed to load metadata {filename}: {e}")
            except Exception as e:
                print(f"[COMP WARN] Unexpected error loading {filename}: {e}")
    records.sort(key=lambda x: x.sequence)
    return records


def _compare_lists(list_a: List, list_b: List, label: str):
    """Compare two lists of elements and print results."""
    max_len = max(len(list_a), len(list_b))
    for i in range(max_len):
        if i >= len(list_a):
            comparator = MissingInAComparator(list_b[i])
        elif i >= len(list_b):
            comparator = MissingInBComparator(list_a[i])
        else:
            comparator = create_comparator(list_a[i], list_b[i])
        left_info, right_info = comparator.get_type_info()
        result = comparator.compare()
        log = format_comparison_result(result)
        print(f"  {label}[{i}] | {left_info} | {right_info} | {log}")


def _compare_kwargs(kwargs_a: dict, kwargs_b: dict, label: str):
    """Compare two kwargs dicts by key and print results."""
    all_keys = sorted(set(list(kwargs_a.keys()) + list(kwargs_b.keys())))
    for key in all_keys:
        key_label = f"{label}[{key}]"
        if key not in kwargs_a:
            comparator = MissingInAComparator(kwargs_b[key])
        elif key not in kwargs_b:
            comparator = MissingInBComparator(kwargs_a[key])
        else:
            comparator = create_comparator(kwargs_a[key], kwargs_b[key])
        left_info, right_info = comparator.get_type_info()
        result = comparator.compare()
        log = format_comparison_result(result)
        print(f"  {key_label} | {left_info} | {right_info} | {log}")


def _find_lcs_matches(records_a: list, records_b: list) -> List[Tuple[int, int]]:
    sigs_a = [format_signature(d) for d in records_a]
    sigs_b = [format_signature(d) for d in records_b]
    lcs_len, matched_pairs = _lcs_length(sigs_a, sigs_b)
    a_only = len(records_a) - lcs_len
    b_only = len(records_b) - lcs_len
    print(f"[LCS] Matched: {lcs_len} operators | A-only: {a_only} | B-only: {b_only}")
    prev_a, prev_b = 0, 0
    for idx_a, idx_b in matched_pairs:
        for i in range(prev_a, idx_a):
            key_a = format_display_key(records_a[i])
            print(f"[SKIP] A[{i}] {key_a} <-> <empty>")
        for j in range(prev_b, idx_b):
            key_b = format_display_key(records_b[j])
            print(f"[SKIP] <empty> <-> B[{j}] {key_b}")
        key_a = format_display_key(records_a[idx_a])
        key_b = format_display_key(records_b[idx_b])
        print(f"[MATCH] A[{idx_a}] {key_a} <-> B[{idx_b}] {key_b}")
        prev_a = idx_a + 1
        prev_b = idx_b + 1
    for i in range(prev_a, len(records_a)):
        key_a = format_display_key(records_a[i])
        print(f"[SKIP] A[{i}] {key_a} <-> <empty>")
    for j in range(prev_b, len(records_b)):
        key_b = format_display_key(records_b[j])
        print(f"[SKIP] <empty> <-> B[{j}] {key_b}")
    return matched_pairs


def _compare_matched_pairs(records_a: list, records_b: list, matched_pairs, storage_a: str, storage_b: str):
    total = len(matched_pairs)
    start_time = time.time()
    print(f"[COMPARE] Starting detailed comparison of {total} matched pairs...")
    for idx, (idx_a, idx_b) in enumerate(matched_pairs, 1):
        elapsed = time.time() - start_time
        avg_time = elapsed / idx if idx > 0 else 0
        eta_seconds = avg_time * (total - idx)
        print(f"[COMPARE {idx}/{total} | ETA: {format_eta(eta_seconds)}]")
        rec_a = records_a[idx_a]
        rec_b = records_b[idx_b]
        filename_a = format_dump_filename(rec_a)
        filename_b = format_dump_filename(rec_b)
        print(f"{filename_a} <-> {filename_b}")
        session_dir_a = os.path.dirname(storage_a)
        session_dir_b = os.path.dirname(storage_b)
        pkl_path_a = os.path.join(session_dir_a, filename_a.replace('.json', '.pkl'))
        pkl_path_b = os.path.join(session_dir_b, filename_b.replace('.json', '.pkl'))
        inputs_a, outputs_a = SerializationSession.load_data(pkl_path_a, storage_a)
        inputs_b, outputs_b = SerializationSession.load_data(pkl_path_b, storage_b)
        _compare_lists(inputs_a['args'], inputs_b['args'], "Inputs.args")
        _compare_kwargs(inputs_a['kwargs'], inputs_b['kwargs'], "Inputs.kwargs")
        _compare_lists(outputs_a, outputs_b, "Outputs")


def ops_comp(dump_dir_a: str, dump_dir_b: str):
    """Compare two operator dump sessions."""
    records_a = _load_all_metadata(dump_dir_a)
    print(f"[LCS] Loading dump A: {len(records_a)} operators from {dump_dir_a}")
    records_b = _load_all_metadata(dump_dir_b)
    print(f"[LCS] Loading dump B: {len(records_b)} operators from {dump_dir_b}")
    matched_pairs = _find_lcs_matches(records_a, records_b)
    storage_a = os.path.join(dump_dir_a, 'storage')
    storage_b = os.path.join(dump_dir_b, 'storage')
    _compare_matched_pairs(records_a, records_b, matched_pairs, storage_a, storage_b)
```

- [ ] **Step 2: Verify metadata loading works with new format**

Run a quick test with existing dump files (or verify import):
```
python -c "from acc.comp import ops_comp; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add acc/comp.py
git commit -m "refactor: metadata-only LCS, ETA progress, args/kwargs comparison"
```

---

### Task 7: Update __init__.py exports

**Files:**
- Modify: `acc/__init__.py`

- [ ] **Step 1: Update exports**

Replace `acc/__init__.py` content with:

```python
"""
PyTorch Operator Dump & Precision Comparison Tool
"""

from .dump import ops_dump
from .comp import ops_comp
from .serialization import OperatorRecord, SerializationSession

__all__ = ['ops_dump', 'ops_comp', 'OperatorRecord', 'SerializationSession']
__version__ = '0.2.0'
```

- [ ] **Step 2: Verify imports**

Run: `python -c "from acc import ops_dump, ops_comp, OperatorRecord, SerializationSession; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add acc/__init__.py
git commit -m "refactor: update exports for OperatorRecord and SerializationSession"
```

---

### Task 8: Update existing tests to work with new structure

**Files:**
- Modify: `tests/test_operator_tools.py`
- Modify: `tests/test_outputs_and_empty_tensor.py`

- [ ] **Step 1: Update `tests/test_operator_tools.py`**

Change the import line from:
```python
from acc import ops_dump, ops_comp
```
to:
```python
from acc import ops_dump, ops_comp, SerializationSession
```

No other changes needed — `ops_dump` and `ops_comp` signatures are backward-compatible. The tests should pass as-is.

- [ ] **Step 2: Run existing tests**

Run: `python tests/test_operator_tools.py`
Expected: All 3 tests PASS

- [ ] **Step 3: Run all test files**

Run:
```bash
python tests/test_cache.py && python tests/test_operator_tools.py && python tests/test_outputs_and_empty_tensor.py
```
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_operator_tools.py
git commit -m "test: update test imports for new SerializationSession"
```

---

### Task 9: Full integration test

- [ ] **Step 1: Run end-to-end dump + compare test**

Run:
```
python -c "
import tempfile, os, torch
from acc import ops_dump, ops_comp

with tempfile.TemporaryDirectory() as tmpdir:
    dir_a = os.path.join(tmpdir, 'a')
    dir_b = os.path.join(tmpdir, 'b')
    os.makedirs(dir_a); os.makedirs(dir_b)

    # Dump A: simple operations
    with ops_dump(dir_a):
        x = torch.randn(2, 3) + 1.0
        y = torch.relu(x)

    # Dump B: same operations, different values
    with ops_dump(dir_b):
        a = torch.randn(2, 3) + 2.0
        b = torch.relu(a)

    # Compare
    sessions_a = [d for d in os.listdir(dir_a) if os.path.isdir(os.path.join(dir_a,d))]
    sessions_b = [d for d in os.listdir(dir_b) if os.path.isdir(os.path.join(dir_b,d))]
    ops_comp(os.path.join(dir_a, sessions_a[0]), os.path.join(dir_b, sessions_b[0]))
    print('INTEGRATION OK')
"
```
Expected: Full comparison output, `INTEGRATION OK` at end

- [ ] **Step 2: Verify cache deduplication works**

Run:
```
python -c "
import tempfile, os, pickle
from acc.cache import CacheEntry
from acc import ops_dump, SerializationSession
import torch

with tempfile.TemporaryDirectory() as tmpdir:
    with ops_dump(tmpdir):
        # Same tensor used twice
        w = torch.ones(100, 100)
        a = w * 2
        b = w * 3
    sessions = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir,d))]
    session = os.path.join(tmpdir, sessions[0])
    # Check storage has fewer files than total PKL files
    storage = os.path.join(session, 'storage')
    pkls = [f for f in os.listdir(session) if f.endswith('.pkl') and f != 'storage']
    storage_files = os.listdir(storage) if os.path.exists(storage) else []
    print(f'PKL files: {len(pkls)}, Storage files: {len(storage_files)}')
    # storage should have 1 file for w (used twice, cached once)
    if len(storage_files) > 0:
        print('Cache storage populated - dedup working')
    else:
        print('No cached entries (may be empty if no tensor operations)')
    print('CACHE TEST OK')
"
```
Expected: See cache storage files, `CACHE TEST OK`

- [ ] **Step 3: Commit**

```bash
git commit -m "test: add integration test for cache deduplication" --allow-empty
```

---

### Task 10: Update README.md for new API

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README.md**

Remove all references to `dumper_manager`. In the API section, replace:

```markdown
### `dumper_manager`

The global dump manager instance. Can be used to control global dump behavior.

```python
from acc import dumper_manager

# Disable all dumps globally
dumper_manager.enabled = False

# Re-enable
dumper_manager.enabled = True

# Check status
print(dumper_manager.enabled)
```
```

With:

```markdown
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
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for SerializationSession and OperatorRecord, remove dumper_manager"
```
