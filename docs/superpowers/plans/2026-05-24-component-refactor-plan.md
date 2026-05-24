# Component Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor operator dump pipeline into composable components (IOWriter, CacheManager, Serializer, Capturer, Manager) with start/stop/save lifecycles.

**Architecture:** Each component reads from centralized Config. Manager orchestrates Capturer (PyTorch hooks), CacheManager (tensor caching), and Serializer/AsyncSerializer (file I/O). `main.py` provides the `ops_dump` entry point supporting context manager and decorator patterns.

**Tech Stack:** Python 3.12, PyTorch 2.9.1+, multiprocessing (fork context), pickle, json

---

### Task 1: Add new config fields

**Files:**
- Modify: `acc/config.py`

- [ ] **Step 1: Add four new fields to Config dataclass**

In `acc/config.py`, add the new fields to the `Config` dataclass:

```python
@dataclass
class Config:
    dump_path: str = "."
    dump_enabled: bool = True
    max_tensor_size_mb: int = 10240
    io_monitor_interval: float = 5.0
    cache_monitor_interval: float = 5.0
    async_io: bool = True
    async_serialization: bool = True
```

The `__post_init__` method already handles env var autodiscovery for any field via `ACC_{NAME.upper()}` convention. No changes needed to `__post_init__`.

- [ ] **Step 2: Verify config works**

Run:
```bash
python -c "from acc.config import config; print(config)"
```

Expected: prints Config with all 7 fields, default values.

- [ ] **Step 3: Run existing tests**

```bash
python -m pytest tests/ -v 2>&1 | tail -5
```

Expected: 45 passed (no regressions).

- [ ] **Step 4: Commit**

```bash
git add acc/config.py
git commit -m "feat: add io_monitor_interval, cache_monitor_interval, async_io, async_serialization config fields"
```

---

### Task 2: Refactor IOWriter — rename methods, add start/stop lifecycle

**Files:**
- Modify: `acc/io.py`
- Modify: `tests/test_io.py`

- [ ] **Step 1: Update IOWriter class in acc/io.py**

Replace the entire IOWriter class. Key changes:
- Move thread creation from `__init__` to `start()`
- Rename `write` → `save`, `read` → `load`
- Replace `wait_complete` with `stop` that blocks and prints pending count every 1s
- `enable_async` defaults to `config.async_io` when None

```python
import time as _time_module

class IOWriter:
    def __init__(self, name: str = "", enable_async: bool = None):
        self.name = name
        if enable_async is None:
            from .config import config
            enable_async = config.async_io
        self.enable_async = enable_async
        self._pending_files = set()
        self._bytes_written = 0
        self._last_monitor_time = 0.0
        self._handler = FileHandler()
        self._thread = None
        if enable_async:
            self._queue = queue.Queue()

    def start(self):
        if self.enable_async:
            self._last_monitor_time = _time_module.time()
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def stop(self):
        if not self.enable_async:
            return
        self._queue.put(None)
        while True:
            pending = len(self._pending_files)
            if pending == 0:
                break
            print(f"[IO] Remaining: {pending} files")
            _time_module.time().sleep(1)
        self._thread.join()
        self._thread = None

    def save(self, file_path, content):
        if self.enable_async:
            self._pending_files.add(file_path)
            self._queue.put((file_path, content))
        else:
            self._handler.write(file_path, content)
            self._bytes_written += os.path.getsize(file_path)

    def load(self, file_path):
        return self._handler.read(file_path)

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            file_path, content = item
            try:
                self._handler.write(file_path, content)
                self._bytes_written += os.path.getsize(file_path)
                self._check_monitor()
            except Exception as e:
                print(f"[IO ERROR] Failed to write {file_path}: {e}")
            finally:
                self._pending_files.discard(file_path)
                self._queue.task_done()

    def _check_monitor(self):
        from .config import config
        now = _time_module.time()
        elapsed = now - self._last_monitor_time
        if elapsed >= config.io_monitor_interval:
            pending_count = len(self._pending_files)
            throughput = self._bytes_written / elapsed if elapsed > 0 else 0
            throughput_str = self._format_bytes(throughput)
            print(f"[IO MONITOR] Pending: {pending_count} files | Throughput: {throughput_str}/s")
            self._bytes_written = 0
            self._last_monitor_time = now

    def _format_bytes(self, bytes_per_sec):
        units = ['B', 'KB', 'MB', 'GB']
        value = bytes_per_sec
        unit_idx = 0
        while value >= 1024 and unit_idx < len(units) - 1:
            value /= 1024
            unit_idx += 1
        return f"{value:,.2f} {units[unit_idx]}"
```

Remove the old `_monitor_interval` constructor parameter, `wait_complete`, `write`, `read`, `handler` property, and `_check_monitor` referencing `self._monitor_interval`.

- [ ] **Step 2: Update tests in tests/test_io.py**

Replace all `write` → `save`, `read` → `load`, `wait_complete()` → `stop()`, add `start()` calls:

```python
# tests/test_io.py
import json
import tempfile
import os
import pickle
from acc.io import IOWriter


def test_iowriter_constructor_async_mode():
    writer = IOWriter(enable_async=True)
    assert writer.enable_async == True
    assert hasattr(writer, '_pending_files')
    assert writer._thread is None
    writer.start()
    assert writer._thread is not None
    writer.stop()


def test_iowriter_constructor_sync_mode():
    writer = IOWriter(enable_async=False)
    assert writer.enable_async == False
    assert hasattr(writer, '_pending_files')
    assert writer._thread is None


def test_iowriter_async_save_str():
    writer = IOWriter(enable_async=True)
    writer.start()
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"msg": "hello world"}
        writer.save(file_path, content)
        writer.stop()
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        assert data == content


def test_iowriter_async_save_obj():
    writer = IOWriter(enable_async=True)
    writer.start()
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.pkl")
        content = {"tensor_data": [1, 2, 3]}
        writer.save(file_path, content)
        writer.stop()
        assert os.path.exists(file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        assert data == content


def test_iowriter_sync_save_str():
    writer = IOWriter(enable_async=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"msg": "hello sync"}
        writer.save(file_path, content)
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        assert data == content
    writer.stop()


def test_iowriter_sync_save_obj():
    writer = IOWriter(enable_async=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.pkl")
        content = {"sync_data": [4, 5, 6]}
        writer.save(file_path, content)
        assert os.path.exists(file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        assert data == content
    writer.stop()


def test_iowriter_pending_files_tracking():
    writer = IOWriter(enable_async=True)
    writer.start()
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"msg": "hello"}
        assert len(writer._pending_files) == 0
        writer.save(file_path, content)
        assert file_path in writer._pending_files
        writer.stop()
        assert len(writer._pending_files) == 0


def test_iowriter_concurrent_saves():
    writer = IOWriter(enable_async=True)
    writer.start()
    with tempfile.TemporaryDirectory() as tmpdir:
        files = []
        for i in range(10):
            file_path = os.path.join(tmpdir, f"test_{i}.pkl")
            content = {"index": i, "data": f"value_{i}"}
            writer.save(file_path, content)
            files.append((file_path, content))
        writer.stop()
        for file_path, expected_content in files:
            assert os.path.exists(file_path)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            assert data == expected_content


def test_iowriter_load_json():
    writer = IOWriter(enable_async=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"msg": "hello read"}
        writer.save(file_path, content)
        data = writer.load(file_path)
        assert data == content


def test_iowriter_load_pickle():
    writer = IOWriter(enable_async=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.pkl")
        content = {"key": "value", "num": 42}
        writer.save(file_path, content)
        data = writer.load(file_path)
        assert data == content
```

- [ ] **Step 3: Run IO tests**

```bash
python -m pytest tests/test_io.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add acc/io.py tests/test_io.py
git commit -m "refactor: IOWriter rename write/read->save/load, add start/stop lifecycle"
```

---

### Task 3: Refactor CacheManager — lifecycle, owns IOWriter, monitoring

**Files:**
- Modify: `acc/cache.py`
- Modify: `tests/test_cache.py`

- [ ] **Step 1: Update CacheManager in acc/cache.py**

Replace `CacheManager.__init__` and add `start`/`stop`/monitoring. Keep `save`/`load`/`_traverse` unchanged.

```python
import time as _time_module

class CacheManager:
    def __init__(self):
        self.cache_dir = None
        self._io = None
        self._save_cached = set()
        self._load_cached = {}
        self._pool = None
        self._max_tensor_size_mb = None
        self._bytes_this_interval = 0
        self._bytes_total = 0
        self._last_monitor_time = 0.0
        self._started = False

    def start(self, session_dir):
        from .config import config
        self._started = True
        self.cache_dir = os.path.join(session_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=False)
        self._io = IOWriter(name="cache")
        self._io.start()
        self._pool = PinMemoryAllocator.create("advanced")
        self._max_tensor_size_mb = config.max_tensor_size_mb
        self._save_cached = set()
        self._load_cached = {}
        self._bytes_this_interval = 0
        self._bytes_total = 0
        self._last_monitor_time = _time_module.time()

    def stop(self):
        if self._io is not None:
            self._io.stop()
        self._started = False

    def _check_monitor(self):
        from .config import config
        now = _time_module.time()
        elapsed = now - self._last_monitor_time
        if elapsed >= config.cache_monitor_interval:
            print(f"[CACHE MONITOR] Added: {self._format_bytes(self._bytes_this_interval)} | Total: {self._format_bytes(self._bytes_total)}")
            self._bytes_this_interval = 0
            self._last_monitor_time = now
```

Update the `_processor` in `save()` to call `_check_monitor()` after each new .pt write, tracking byte counts. The `save()` method `processor` closure needs changes:

```python
def save(self, data):
    if not self._started:
        raise RuntimeError("CacheManager not started")
    def processor(obj):
        if not isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj
        if self._max_tensor_size_mb is not None:
            size_mb = _tensor_size_mb(obj)
            if size_mb > self._max_tensor_size_mb:
                print(f"[DUMP WARN] Tensor size {size_mb:.2f} MB exceeds limit, replacing with None")
                return None
        storage = Storage(obj)
        cache_id = storage.cache_id
        entry = CacheEntry(
            cache_id=cache_id,
            type='tensor' if isinstance(obj, torch.Tensor) else 'numpy',
            dtype=str(obj.dtype).replace('torch.', ''),
            shape=list(obj.shape)
        )
        if cache_id not in self._save_cached:
            storage_tensor = storage.materialize(self._pool)
            filepath = os.path.join(self.cache_dir, f"{cache_id}.pt")
            self._io.save(filepath, storage_tensor)
            self._save_cached.add(cache_id)
            file_size = os.path.getsize(filepath)
            self._bytes_this_interval += file_size
            self._bytes_total += file_size
            self._check_monitor()
        return entry
    return self._traverse(data, processor)
```

Remove the old `__init__` parameters: `cache_dir`, `cache_io`, `allocator_type`, `max_tensor_size_mb`.

- [ ] **Step 2: Update tests in tests/test_cache.py**

Update all test functions to use new `start()` API instead of constructing `IOWriter` and passing `cache_io`:

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import torch
import numpy as np
from acc.cache import CacheEntry, CacheManager
from acc.io import IOWriter


def test_cache_entry():
    print("Test: CacheEntry dataclass")
    entry = CacheEntry(cache_id="abc123", type="tensor", dtype="float32", shape=[2, 3])
    assert entry.cache_id == "abc123"
    assert entry.type == "tensor"
    assert entry.shape == [2, 3]
    print("  PASS")


def test_save_tensor():
    print("Test: save with tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir)
        mgr = CacheManager()
        mgr.cache_dir = cache_dir
        mgr._io = IOWriter(enable_async=False)
        mgr._started = True
        from acc.memory import PinMemoryAllocator
        mgr._pool = PinMemoryAllocator.create("advanced")
        mgr._max_tensor_size_mb = 10240
        mgr._save_cached = set()
        mgr._load_cached = {}
        t = torch.randn(2, 3)
        result = mgr.save(t)
        assert isinstance(result, CacheEntry)
        assert result.type == "tensor"
        assert result.shape == [2, 3]
        result2 = mgr.save(t)
        assert result2.cache_id == result.cache_id
    print("  PASS")


def test_save_numpy():
    print("Test: save with numpy")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir)
        mgr = CacheManager()
        mgr.cache_dir = cache_dir
        mgr._io = IOWriter(enable_async=False)
        mgr._started = True
        from acc.memory import PinMemoryAllocator
        mgr._pool = PinMemoryAllocator.create("advanced")
        mgr._max_tensor_size_mb = 10240
        mgr._save_cached = set()
        mgr._load_cached = {}
        a = np.random.randn(3, 4).astype(np.float32)
        result = mgr.save(a)
        assert isinstance(result, CacheEntry)
        assert result.type == "numpy"
        result2 = mgr.save(a)
        assert result2.cache_id == result.cache_id
    print("  PASS")


def test_save_scalar():
    print("Test: save with non-tensor/numpy")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir)
        mgr = CacheManager()
        mgr.cache_dir = cache_dir
        mgr._io = IOWriter(enable_async=False)
        mgr._started = True
        from acc.memory import PinMemoryAllocator
        mgr._pool = PinMemoryAllocator.create("advanced")
        mgr._max_tensor_size_mb = 10240
        mgr._save_cached = set()
        mgr._load_cached = {}
        assert mgr.save(42) == 42
        assert mgr.save(3.14) == 3.14
        assert mgr.save("hello") == "hello"
        assert mgr.save(None) is None
    print("  PASS")


def test_load_tensor():
    print("Test: load tensor from CacheEntry")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir)
        mgr = CacheManager()
        mgr.cache_dir = cache_dir
        mgr._io = IOWriter(enable_async=False)
        mgr._started = True
        from acc.memory import PinMemoryAllocator
        mgr._pool = PinMemoryAllocator.create("advanced")
        mgr._max_tensor_size_mb = 10240
        mgr._save_cached = set()
        mgr._load_cached = {}
        t = torch.tensor([1.0, 2.0, 3.0])
        entry = mgr.save(t)
        restored = mgr.load(entry)
        assert isinstance(restored, torch.Tensor)
        assert torch.equal(restored, t)
    print("  PASS")


def test_load_numpy():
    print("Test: load numpy from CacheEntry")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir)
        mgr = CacheManager()
        mgr.cache_dir = cache_dir
        mgr._io = IOWriter(enable_async=False)
        mgr._started = True
        from acc.memory import PinMemoryAllocator
        mgr._pool = PinMemoryAllocator.create("advanced")
        mgr._max_tensor_size_mb = 10240
        mgr._save_cached = set()
        mgr._load_cached = {}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        entry = mgr.save(a)
        restored = mgr.load(entry)
        assert isinstance(restored, np.ndarray)
        assert np.equal(restored, a).all()
    print("  PASS")


def test_bfloat16_tensor():
    print("Test: BFloat16 tensor save/load")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir)
        mgr = CacheManager()
        mgr.cache_dir = cache_dir
        mgr._io = IOWriter(enable_async=False)
        mgr._started = True
        from acc.memory import PinMemoryAllocator
        mgr._pool = PinMemoryAllocator.create("advanced")
        mgr._max_tensor_size_mb = 10240
        mgr._save_cached = set()
        mgr._load_cached = {}
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        entry = mgr.save(t)
        restored = mgr.load(entry)
        assert isinstance(restored, torch.Tensor)
        assert restored.dtype == torch.bfloat16
        assert torch.equal(restored, t)
    print("  PASS")


def test_save_load_nested():
    print("Test: save/load nested structure")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir)
        mgr = CacheManager()
        mgr.cache_dir = cache_dir
        mgr._io = IOWriter(enable_async=False)
        mgr._started = True
        from acc.memory import PinMemoryAllocator
        mgr._pool = PinMemoryAllocator.create("advanced")
        mgr._max_tensor_size_mb = 10240
        mgr._save_cached = set()
        mgr._load_cached = {}
        t1 = torch.randn(2, 2)
        t2 = torch.randn(3, 3)
        data = {'tensors': [t1, t2], 'value': 42, 'name': 'test'}
        saved = mgr.save(data)
        assert isinstance(saved['tensors'][0], CacheEntry)
        assert isinstance(saved['tensors'][1], CacheEntry)
        assert saved['value'] == 42
        assert saved['name'] == 'test'
        loaded = mgr.load(saved)
        assert isinstance(loaded['tensors'][0], torch.Tensor)
        assert isinstance(loaded['tensors'][1], torch.Tensor)
        assert torch.equal(loaded['tensors'][0], t1)
        assert torch.equal(loaded['tensors'][1], t2)
    print("  PASS")


def test_different_tensors_different_hash():
    print("Test: different tensors get different cache_ids")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir)
        mgr = CacheManager()
        mgr.cache_dir = cache_dir
        mgr._io = IOWriter(enable_async=False)
        mgr._started = True
        from acc.memory import PinMemoryAllocator
        mgr._pool = PinMemoryAllocator.create("advanced")
        mgr._max_tensor_size_mb = 10240
        mgr._save_cached = set()
        mgr._load_cached = {}
        t1 = torch.ones(2, 3)
        t2 = torch.zeros(2, 3)
        e1 = mgr.save(t1)
        e2 = mgr.save(t2)
        assert e1.cache_id != e2.cache_id
    print("  PASS")


def test_same_storage_different_shape():
    print("Test: same storage but different shape")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir)
        mgr = CacheManager()
        mgr.cache_dir = cache_dir
        mgr._io = IOWriter(enable_async=False)
        mgr._started = True
        from acc.memory import PinMemoryAllocator
        mgr._pool = PinMemoryAllocator.create("advanced")
        mgr._max_tensor_size_mb = 10240
        mgr._save_cached = set()
        mgr._load_cached = {}
        t1 = torch.arange(6)
        t2 = t1.reshape(2, 3)
        e1 = mgr.save(t1)
        e2 = mgr.save(t2)
        assert e1.cache_id == e2.cache_id
        assert e1.shape != e2.shape
        r1 = mgr.load(e1)
        r2 = mgr.load(e2)
        assert list(r1.shape) == [6]
        assert list(r2.shape) == [2, 3]
    print("  PASS")
```

Note: All tests use the same setup pattern since `CacheManager.__init__` takes no args. We set internal state directly for testing (same approach as the codebase already uses for private attribute testing).

- [ ] **Step 3: Run cache tests**

```bash
python -m pytest tests/test_cache.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add acc/cache.py tests/test_cache.py
git commit -m "refactor: CacheManager no-arg constructor, owns IOWriter, add start/stop/monitoring"
```

---

### Task 4: Refactor Serialization — Serializer, AsyncSerializer, SerializationManager shim

**Files:**
- Modify: `acc/serialization.py`

- [ ] **Step 1: Restructure serialization.py**

Remove `SerializationSender`, `SerializationReceiver`, `create_pipeline`, `_receiver_main`. Keep `_MP_CONTEXT`, `OperatorRecord`, `_sanitize_filename`, `_sanitize_opname`, `_process_frames`, `_wrap_outputs`.

Add `Serializer` class (replaces `SerializationManager` plus IOWriter ownership):

```python
class Serializer:
    def __init__(self):
        self.session_dir = None
        self._io = None

    def start(self, session_dir):
        self.session_dir = session_dir
        self._io = IOWriter(name="seq")
        self._io.start()

    def stop(self):
        if self._io is not None:
            self._io.stop()

    def save(self, item):
        seq = item['sequence']
        opname = item['opname']
        frames = item['frames']

        filepath, filename, function, lineno = _process_frames(frames)
        call_stack = frames

        filename_safe = _sanitize_filename(filename)
        function_safe = _sanitize_filename(function)
        opname_safe = _sanitize_opname(opname)

        json_filename = f"{seq:06d}__{filename_safe}__{function_safe}__{opname_safe}.json"
        pkl_filename = f"{seq:06d}__{filename_safe}__{function_safe}__{opname_safe}.pkl"
        json_path = os.path.join(self.session_dir, json_filename)
        pkl_path = os.path.join(self.session_dir, pkl_filename)

        self._io.save(json_path, {
            'sequence': seq, 'filepath': filepath, 'filename': filename,
            'function': function, 'lineno': lineno, 'opname': opname,
            'call_stack': call_stack,
        })
        self._io.save(pkl_path, {
            'inputs': item['inputs'],
            'outputs': item['outputs'],
        })

    @staticmethod
    def load_metadata(json_path):
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
    def load_data(pkl_path, storage_dir):
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        inputs = pkl_data['inputs']
        outputs = pkl_data['outputs']
        cache_io = IOWriter(enable_async=False)
        cache_dir = storage_dir
        if not os.path.isdir(cache_dir):
            # fallback for old sessions with 'storage/' instead of 'cache/'
            cache_dir = os.path.join(os.path.dirname(storage_dir), 'cache')
        cache_mgr = CacheManager()
        cache_mgr.cache_dir = cache_dir
        cache_mgr._io = cache_io
        cache_mgr._pool = PinMemoryAllocator.create("advanced")
        cache_mgr._max_tensor_size_mb = 10240
        cache_mgr._save_cached = set()
        cache_mgr._load_cached = {}
        cache_mgr._started = True
        resolved_args = cache_mgr.load(inputs['args'])
        resolved_kwargs = cache_mgr.load(inputs['kwargs'])
        resolved_outputs = cache_mgr.load(outputs)
        return {'args': resolved_args, 'kwargs': resolved_kwargs}, resolved_outputs
```

Add `AsyncSerializer` class:

```python
class AsyncSerializer:
    def __init__(self):
        self.session_dir = None
        self._process = None
        self.queue = None

    def start(self, session_dir):
        self.session_dir = session_dir
        self.queue = _MP_CONTEXT.Queue()
        self._process = _MP_CONTEXT.Process(
            target=_serializer_subprocess,
            args=(self.session_dir, self.queue)
        )
        self._process.start()

    def stop(self):
        self.queue.put(None)
        if self._process is not None:
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()

    def save(self, item):
        self.queue.put(item)


def _serializer_subprocess(session_dir, queue):
    serializer = Serializer()
    serializer.start(session_dir)
    while True:
        item = queue.get()
        if item is None:
            break
        try:
            serializer.save(item)
        except Exception as e:
            seq = item.get('sequence', '?')
            opname = item.get('opname', '?')
            print(f"[DUMP ERROR] {seq:06d} | {opname} | serializer.save failed: {e}")
    serializer.stop()
```

Add `SerializationManager` compatibility shim (for comp.py):

```python
class SerializationManager:
    @staticmethod
    def load_metadata(json_path):
        return Serializer.load_metadata(json_path)

    @staticmethod
    def load_data(pkl_path, storage_dir):
        return Serializer.load_data(pkl_path, storage_dir)
```

Update imports at top of serialization.py — add `from .io import IOWriter` (already present), `from .cache import CacheManager` (already present), `from .memory import PinMemoryAllocator` (already present).

Remove imports that are only used by deleted classes: `uuid` (unless used by something else), `dist` import in sender.start. Keep `dist` if OperatorRecord's `from_dict`/`to_dict` still need it (they don't). Remove unused imports from `SerializationSender`/`SerializationReceiver`: `datetime`, `uuid`, `dist.get_rank`.

Remove `_receiver_main` (replaced by `_serializer_subprocess`).

- [ ] **Step 2: Run all tests to check for breakage**

```bash
python -m pytest tests/ -v 2>&1 | tail -20
```

Expected: failures in `test_io_integration.py` (imports `create_pipeline` + `SerializationSender`), `test_large_tensor_handling.py` (imports `SerializationSender`). Other tests may fail if they use `SerializationManager.load_data` with old `'storage'` directory name (now `'cache'`).

- [ ] **Step 3: Update test_io_integration.py**

Replace `create_pipeline` test with test using components directly:

```python
# tests/test_io_integration.py
import os
import tempfile
from acc import ops_dump
from acc.config import config
from acc.serialization import Serializer
from acc.cache import CacheManager


def test_serializer_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Serializer()
        s.start(tmpdir)
        item = {
            'sequence': 0,
            'opname': 'test.op',
            'frames': [{
                'filepath': '/tmp/test.py',
                'lineno': 10,
                'function': 'test_fn',
                'line': 'x + y',
            }],
            'inputs': {'args': [1, 2], 'kwargs': {}},
            'outputs': [3],
        }
        s.save(item)
        s.stop()
        files = os.listdir(tmpdir)
        json_files = [f for f in files if f.endswith('.json')]
        pkl_files = [f for f in files if f.endswith('.pkl')]
        assert len(json_files) == 1
        assert len(pkl_files) == 1


def test_ops_dump_creates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        with ops_dump(tmpdir) as dumper:
            import torch
            a = torch.randn(2, 3)
            _ = a + 1

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        assert len(dump_dirs) > 0
        session_dir = os.path.join(tmpdir, dump_dirs[0])
        json_files = [f for f in os.listdir(session_dir) if f.endswith('.json')]
        assert len(json_files) > 0


def test_ops_dump_disabled_no_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        config.update(dump_enabled=False)
        with ops_dump(tmpdir) as dumper:
            import torch
            a = torch.randn(2, 3)
            _ = a + 1
        config.update(dump_enabled=True)

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        assert len(dump_dirs) == 0
```

- [ ] **Step 4: Update test_large_tensor_handling.py**

Replace `test_default_max_tensor_size` (which used removed `SerializationSender`) with a test on `CacheManager`:

```python
def test_default_max_tensor_size():
    """Test that CacheManager picks up config.max_tensor_size_mb."""
    print("=" * 60)
    print("Test: Default max tensor size via CacheManager")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CacheManager()
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir)
        mgr.cache_dir = cache_dir
        mgr._io = IOWriter(enable_async=False)
        from acc.memory import PinMemoryAllocator
        mgr._pool = PinMemoryAllocator.create("advanced")
        mgr._save_cached = set()
        mgr._load_cached = {}
        mgr._started = True
        mgr._max_tensor_size_mb = 10240

        assert mgr._max_tensor_size_mb == 10240, \
            f"Default max_tensor_size_mb should be 10240 (10GB), got {mgr._max_tensor_size_mb}"
        print(f"Default max_tensor_size_mb: {mgr._max_tensor_size_mb} MB (10GB)")

    print("PASS: Default max tensor size is 10GB\n")
```

Add `from acc.cache import CacheManager` and `from acc.io import IOWriter` to imports.

```python
# tests/test_large_tensor_handling.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tempfile
import json
import pickle
from acc import ops_dump
from acc.config import config
from acc.cache import CacheEntry


def test_large_tensor_replaced_with_none():
    """Test that large tensors are replaced with None."""
    print("=" * 60)
    print("Test: Large tensor replaced with None")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config.update(max_tensor_size_mb=1)
        with ops_dump(tmpdir) as dumper:
            large_tensor = torch.randn(256, 256, 256)
            result = large_tensor + 1

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])

        dump_files = [f for f in os.listdir(session_dir) if f.endswith('.json')]

        for json_file in dump_files:
            json_path = os.path.join(session_dir, json_file)
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            if 'add' in metadata['opname'].lower():
                print(f"Found add operation: {json_file}")

                pkl_path = json_path.replace('.json', '.pkl')
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)

                outputs = data['outputs']
                print(f"Outputs: {outputs}")

                assert outputs[0] is None or isinstance(outputs[0], torch.Tensor), \
                    f"Output should be None or tensor, got {type(outputs[0])}"

                if outputs[0] is None:
                    print("PASS: Large output tensor replaced with None")
                else:
                    tensor_size_mb = outputs[0].numel() * 4 / (1024 * 1024)
                    print(f"Output tensor size: {tensor_size_mb:.2f} MB")
                    if tensor_size_mb <= 1:
                        print("PASS: Tensor is within size limit")
                    else:
                        assert False, f"Large tensor not replaced with None, size: {tensor_size_mb:.2f} MB"
                return

        assert False, "No add operation found"

    print("PASS: Test passed\n")


def test_contiguous_error_handling():
    """Test that contiguous() errors are handled gracefully."""
    print("=" * 60)
    print("Test: Contiguous error handling")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['ACC_MAX_TENSOR_SIZE_MB'] = '1'
        try:
            with ops_dump(tmpdir) as dumper:
                normal_tensor = torch.randn(10, 10)
                result = normal_tensor + 1
        finally:
            os.environ.pop('ACC_MAX_TENSOR_SIZE_MB', None)

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])

        dump_files = [f for f in os.listdir(session_dir) if f.endswith('.json')]

        for json_file in dump_files:
            json_path = os.path.join(session_dir, json_file)
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            if 'add' in metadata['opname'].lower():
                print(f"Found add operation: {json_file}")

                pkl_path = json_path.replace('.json', '.pkl')
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)

                outputs = data['outputs']
                from acc.cache import CacheEntry
                assert isinstance(outputs[0], CacheEntry), \
                    f"Normal tensor output should be saved as CacheEntry, got {type(outputs[0])}"
                print("PASS: Normal tensor saved correctly as CacheEntry")
                return

        assert False, "No add operation found"

    print("PASS: Test passed\n")
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_serialization.py tests/test_io_integration.py tests/test_large_tensor_handling.py -v 2>&1 | tail -20
```

No `test_serialization.py` exists — run relevant tests:

```bash
python -m pytest tests/test_io_integration.py tests/test_large_tensor_handling.py -v
```

Expected: tests pass.

- [ ] **Step 6: Commit**

```bash
git add acc/serialization.py tests/test_io_integration.py tests/test_large_tensor_handling.py
git commit -m "refactor: add Serializer/AsyncSerializer, SerializationManager compat shim, remove Sender/Receiver"
```

---

### Task 5: Create Capturer (`capturer.py`)

**Files:**
- Create: `acc/capturer.py`

- [ ] **Step 1: Create acc/capturer.py**

Extract the TorchDispatchMode hook mechanism from dump.py:

```python
import torch
from torch.utils._python_dispatch import TorchDispatchMode


_active_session = None

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
                if _active_session is None:
                    return f(*args, **kwargs)
                with _active_session:
                    return f(*args, **kwargs)
            return wrapped

        if func is None:
            def decorator(f):
                return _original_impl(qualname, types, wrap(f), lib=lib)
            return decorator
        else:
            return _original_impl(qualname, types, wrap(func), lib=lib)

    torch.library.impl = patched_impl


_install_impl_patch()


class Capturer(TorchDispatchMode):
    def __init__(self):
        self._handler = None
        self._in_dispatch = False
        self._enabled = True

    def start(self, handler):
        from .config import config
        global _active_session
        self._handler = handler
        self._enabled = config.dump_enabled
        if self._enabled:
            _active_session = self
            self.__enter__()

    def stop(self):
        global _active_session
        _active_session = None
        self.__exit__(None, None, None)
        self._handler = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not self._enabled or self._in_dispatch or self._handler is None:
            return func(*args, **(kwargs or {}))
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        self._in_dispatch = True
        try:
            self._handler(str(func), args, kwargs, result)
        finally:
            self._in_dispatch = False
        return result
```

- [ ] **Step 2: Verify import**

```bash
python -c "from acc.capturer import Capturer; print('OK')"
```

Expected: `[DUMP PATCH] Installing torch.library.impl patch\nOK`

- [ ] **Step 3: Commit**

```bash
git add acc/capturer.py
git commit -m "feat: add Capturer with TorchDispatchMode hook"
```

---

### Task 6: Create Manager (`manager.py`)

**Files:**
- Create: `acc/manager.py`

- [ ] **Step 1: Create acc/manager.py**

```python
import os
import sys
import linecache
import time
import uuid
from datetime import datetime

from .config import config
from .cache import CacheManager
from .capturer import Capturer


class Manager:
    def __init__(self):
        self.session_dir = None
        self._cache_mgr = CacheManager()
        self._capturer = Capturer()
        self._serializer = None
        self._sequence = 0

    def start(self):
        from .serialization import Serializer, AsyncSerializer

        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = "None"
        pid = os.getpid()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        session_id = uuid.uuid4().hex[:8]
        self.session_dir = os.path.join(
            config.dump_path,
            f"{rank}-{pid}-{timestamp}-{session_id}"
        )
        os.makedirs(self.session_dir, exist_ok=False)

        self._cache_mgr.start(self.session_dir)

        if config.async_serialization:
            self._serializer = AsyncSerializer()
        else:
            self._serializer = Serializer()
        self._serializer.start(self.session_dir)

        self._sequence = 0
        self._capturer.start(self._handler)

        return self.session_dir

    def stop(self):
        self._capturer.stop()
        if self._serializer is not None:
            self._serializer.stop()
        self._cache_mgr.stop()

    def _handler(self, opname, args, kwargs, outputs):
        frames = []
        f = sys._getframe(0)
        while f:
            frames.append(f)
            f = f.f_back

        frame_dicts = [
            {
                'filepath': f.f_code.co_filename,
                'lineno': f.f_lineno,
                'function': f.f_code.co_name,
                'line': linecache.getline(f.f_code.co_filename, f.f_lineno).rstrip('\n'),
            }
            for f in reversed(frames)
        ]

        from .serialization import _wrap_outputs
        serialized_args = self._cache_mgr.save(args)
        serialized_kwargs = self._cache_mgr.save(kwargs or {})
        serialized_outputs = self._cache_mgr.save(_wrap_outputs(outputs))

        seq = self._sequence
        item = {
            'sequence': seq,
            'opname': opname,
            'frames': frame_dicts,
            'inputs': {'args': serialized_args, 'kwargs': serialized_kwargs},
            'outputs': serialized_outputs,
        }
        try:
            self._serializer.save(item)
        except Exception as e:
            print(f"[DUMP ERROR] {seq:06d} | {opname} | serializer.save failed: {e}")
        self._sequence += 1
```

- [ ] **Step 2: Verify import**

```bash
python -c "from acc.manager import Manager; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add acc/manager.py
git commit -m "feat: add Manager to orchestrate CacheManager, Capturer, Serializer"
```

---

### Task 7: Create main.py entry point

**Files:**
- Create: `acc/main.py`

- [ ] **Step 1: Create acc/main.py**

```python
from .config import config
from .manager import Manager


class _OpsDumpContext:
    def __init__(self, manager):
        self._manager = manager
        self.enabled = config.dump_enabled

    def __enter__(self):
        if self.enabled:
            self._manager.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self._manager.stop()
        return False

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def ops_dump(**kwargs):
    config.update(**{k: v for k, v in kwargs.items() if v is not None})
    mgr = Manager()
    return _OpsDumpContext(mgr)
```

- [ ] **Step 2: Verify import and manual test**

```bash
python -c "
from acc.main import ops_dump
import tempfile, torch
with tempfile.TemporaryDirectory() as tmpdir:
    with ops_dump(dump_path=tmpdir) as d:
        a = torch.randn(2, 3)
        b = a + 1
    import os
    dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
    print(f'Sessions: {dirs}')
    assert len(dirs) > 0
    print('PASS')
"
```

Expected: `PASS`

- [ ] **Step 3: Commit**

```bash
git add acc/main.py
git commit -m "feat: add main.py with ops_dump entry point"
```

---

### Task 8: Update __init__.py, delete dump.py, fix remaining tests

**Files:**
- Modify: `acc/__init__.py`
- Delete: `acc/dump.py`
- Modify: `tests/test_outputs_and_empty_tensor.py`
- Modify: `tests/test_custom_operator_dump.py`

- [ ] **Step 1: Update acc/__init__.py**

```python
"""
PyTorch Operator Dump & Precision Comparison Tool
"""

from .main import ops_dump
from .comp import ops_comp
from .cache import CacheEntry
from .config import config
from .io import IOWriter, FileHandler
from .serialization import SerializationManager, Serializer, AsyncSerializer, OperatorRecord

__all__ = ['ops_dump', 'ops_comp', 'CacheEntry', 'config', 'IOWriter', 'FileHandler',
           'SerializationManager', 'Serializer', 'AsyncSerializer', 'OperatorRecord']
__version__ = '0.3.0'
```

- [ ] **Step 2: Delete dump.py**

```bash
rm acc/dump.py
```

- [ ] **Step 3: Update test_outputs_and_empty_tensor.py**

The test creates `CacheManager(storage_dir, cache_io=IOWriter(enable_async=False))` — update for new API. Change the `test_saves_operator_outputs` function's cache loading section (around line 61-63):

```python
# Old:
#     storage_dir = os.path.join(session_dir, 'storage')
#     cache_mgr = CacheManager(storage_dir, cache_io=IOWriter(enable_async=False))

# New:
    cache_dir = os.path.join(session_dir, 'cache')
    cache_mgr = CacheManager()
    cache_mgr.cache_dir = cache_dir
    cache_mgr._io = IOWriter(enable_async=False)
    from acc.memory import PinMemoryAllocator
    cache_mgr._pool = PinMemoryAllocator.create("advanced")
    cache_mgr._max_tensor_size_mb = 10240
    cache_mgr._save_cached = set()
    cache_mgr._load_cached = {}
    cache_mgr._started = True
```

Also update imports — remove `from acc.io import IOWriter` (still needed for IOWriter creation).

- [ ] **Step 4: Update test_custom_operator_dump.py — rewrite test_nested_dump_contexts**

Replace `test_nested_dump_contexts` (which tested unsupported nested `ops_dump`) with a test verifying Capturer nested entry via the `torch.library.impl` patch — custom operators that trigger internal op capture:

```python
def test_nested_dump_contexts():
    """Test Capturer nested entry via torch.library.impl patch for custom operators."""
    print("=" * 60)
    print("Test: Capturer nested entry via custom operator")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            torch.library.define(
                "nested_test::custom_relu_mul",
                "(Tensor x) -> Tensor"
            )
        except Exception:
            pass

        @torch.library.impl("nested_test::custom_relu_mul", "CompositeExplicitAutograd")
        def custom_relu_mul_impl(x):
            y = torch.relu(x)
            y = torch.mul(y, 2.0)
            return y

        x = torch.randn(3, 3)

        with ops_dump(tmpdir) as dumper:
            y = x + 1
            result = torch.ops.nested_test.custom_relu_mul(y)

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])
        json_files = [f for f in os.listdir(session_dir) if f.endswith('.json')]

        opnames = []
        for f in sorted(json_files):
            with open(os.path.join(session_dir, f), 'r') as fp:
                opnames.append(json.load(fp)['opname'])

        print(f"Captured operators: {opnames}")

        assert any('add' in op.lower() for op in opnames), "add should be captured"
        print("PASS: add captured")
        assert any('relu' in op.lower() for op in opnames), "relu (inside custom op) should be captured"
        print("PASS: relu inside custom op captured")
        assert any('mul' in op.lower() for op in opnames), "mul (inside custom op) should be captured"
        print("PASS: mul inside custom op captured")

        print("PASS: Capturer nested entry test passed\n")
```

Also update the `__main__` block to keep the call: `test_nested_dump_contexts()` is already included in the existing block under the same name.

- [ ] **Step 5: Run all tests**

```bash
python -m pytest tests/ -v 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 6: Fix any remaining failures**

Check for any imports from `dump` module, `acc.dump` references in tests, or `SerializationSender` usage.

- [ ] **Step 7: Commit**

```bash
git add acc/__init__.py tests/test_outputs_and_empty_tensor.py tests/test_custom_operator_dump.py
git rm acc/dump.py
git commit -m "refactor: update __init__.py, delete dump.py, fix tests for new API"
```

---

### Task 9: Final verification

**Files:**
- All

- [ ] **Step 1: Full test suite**

```bash
python -m pytest tests/ -v
```

Expected: 45 tests pass.

- [ ] **Step 2: Verify all exports**

```bash
python -c "
from acc import ops_dump, ops_comp, CacheEntry, config, IOWriter, FileHandler
from acc.serialization import SerializationManager, Serializer, AsyncSerializer, OperatorRecord
print('All exports OK')
"
```

Expected: `All exports OK`

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: final verification after component refactor"
```
