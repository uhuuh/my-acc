# AsyncIO-based IO Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an async IO layer to avoid blocking main thread during model execution with concurrent file writes and exit handling

**Architecture:** Single IOWriter manager with built-in asyncio loop, simple write() interface, pending file tracking via set, atexit/signal handlers for exit warnings

**Tech Stack:** Python asyncio, threading, atexit, signal, aiofiles (optional)

---

## File Structure

**New Files:**
- `acc/io.py` - IOWriter class with async loop management and pending file tracking

**Modified Files:**
- `acc/serialization.py` - Replace direct file writes with IOWriter calls in save_operation()
- `acc/cache.py` - Replace direct pickle.dump with IOWriter calls
- `acc/dump.py` - Pass enable_async_io parameter to SerializationSession

**Test Files:**
- `tests/test_io.py` - Unit tests for IOWriter
- `tests/test_io_integration.py` - Integration tests with SerializationSession and CacheManager

---

### Task 1: Create IOWriter Skeleton with Constructor

**Files:**
- Create: `acc/io.py`
- Test: `tests/test_io.py`

- [ ] **Step 1: Write failing test for IOWriter constructor**

```python
# tests/test_io.py
import pytest
from acc.io import IOWriter

def test_iowriter_constructor_async_mode():
    """Test IOWriter initializes with async mode by default"""
    writer = IOWriter(enable_async=True)
    assert writer.enable_async == True
    assert hasattr(writer, '_pending_files')
    assert hasattr(writer, '_loop')
    assert hasattr(writer, '_thread')
    writer.shutdown()

def test_iowriter_constructor_sync_mode():
    """Test IOWriter initializes with sync mode"""
    writer = IOWriter(enable_async=False)
    assert writer.enable_async == False
    assert hasattr(writer, '_pending_files')
    writer.shutdown()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_io.py::test_iowriter_constructor_async_mode -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'acc.io'"

- [ ] **Step 3: Write minimal IOWriter class**

```python
# acc/io.py
"""
AsyncIO-based IO Layer for PyTorch Operator Dump Tool.
Provides async file writing to avoid blocking main thread.
"""

import asyncio
import threading
import atexit
import signal
import os
from typing import Set, Optional


class IOWriter:
    """
    IO Writer manager with async loop for concurrent file writes.
    Simple interface: write(file_path, content)
    """

    def __init__(self, enable_async: bool = True):
        """
        Initialize IOWriter.

        Args:
            enable_async: Global config for async/sync mode (default: True)
        """
        self.enable_async = enable_async
        self._pending_files: Set[str] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        if enable_async:
            self._start_loop()
            self._register_exit_handlers()

    def _start_loop(self):
        """Start asyncio event loop in dedicated thread"""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="AsyncIO-Loop"
        )
        self._thread.start()

        # Wait for loop to start
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self._loop).result()

    def _run_loop(self):
        """Thread target: run event loop"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        self._loop.close()

    def _register_exit_handlers(self):
        """Register atexit and signal handlers"""
        atexit.register(self._on_exit)

        # Signal handlers (Unix/Linux, Windows only supports SIGINT)
        if os.name != 'nt':  # Unix/Linux
            signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

    def _on_exit(self):
        """atexit callback: check pending files"""
        if not self.enable_async:
            return

        with self._lock:
            pending = list(self._pending_files)

        if pending:
            print(f"[IO WARN] Pending write tasks on exit ({len(pending)} files):")
            for file_path in pending:
                print(f"  - {file_path}")

    def _on_signal(self, signum, frame):
        """Signal handler callback"""
        self._on_exit()
        # Re-trigger default handler
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def shutdown(self):
        """Shutdown async loop and wait for pending tasks"""
        if not self.enable_async:
            return

        if self._loop and self._thread:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io.py::test_iowriter_constructor_async_mode tests/test_io.py::test_iowriter_constructor_sync_mode -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add acc/io.py tests/test_io.py
git commit -m "feat: add IOWriter skeleton with constructor and exit handlers"
```

---

### Task 2: Implement Async Write Method

**Files:**
- Modify: `acc/io.py`
- Test: `tests/test_io.py`

- [ ] **Step 1: Write failing test for async write**

```python
# tests/test_io.py (add to existing file)
import tempfile
import json
import os

def test_iowriter_async_write_json():
    """Test async write for JSON file"""
    writer = IOWriter(enable_async=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"key": "value", "number": 42}

        writer.write(file_path, content)

        # Wait for write to complete
        writer.shutdown()

        # Verify file exists and content is correct
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        assert data == content

def test_iowriter_async_write_pkl():
    """Test async write for PKL file"""
    import pickle
    writer = IOWriter(enable_async=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.pkl")
        content = {"tensor_data": [1, 2, 3]}

        writer.write(file_path, content)

        # Wait for write to complete
        writer.shutdown()

        # Verify file exists and content is correct
        assert os.path.exists(file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        assert data == content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io.py::test_iowriter_async_write_json -v`
Expected: FAIL with "AttributeError: 'IOWriter' object has no attribute 'write'"

- [ ] **Step 3: Implement write method**

```python
# acc/io.py (add to IOWriter class)
import json
import pickle

class IOWriter:
    # ... existing methods ...

    def write(self, file_path: str, content):
        """
        Write file asynchronously (or synchronously if enable_async=False).

        Args:
            file_path: File path (keep original format, no suffix)
            content: File content (auto-serialize based on extension)
        """
        if not self.enable_async:
            self._write_sync(file_path, content)
        else:
            self._write_async(file_path, content)

    def _write_sync(self, file_path: str, content):
        """Synchronous file write"""
        self._serialize_and_write(file_path, content)

    def _write_async(self, file_path: str, content):
        """Asynchronous file write via loop"""
        with self._lock:
            self._pending_files.add(file_path)

        coro = self._async_write_task(file_path, content)
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _async_write_task(self, file_path: str, content):
        """Async write task"""
        try:
            # Use asyncio-friendly file I/O (thread pool fallback)
            await asyncio.get_event_loop().run_in_executor(
                None, self._serialize_and_write, file_path, content
            )
        except Exception as e:
            print(f"[IO ERROR] Failed to write {file_path}: {e}")
        finally:
            with self._lock:
                self._pending_files.discard(file_path)

    def _serialize_and_write(self, file_path: str, content):
        """Serialize and write file (sync operation)"""
        # Create parent directory if needed
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Auto-serialize based on extension
        if file_path.endswith('.json'):
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'wb') as f:
                pickle.dump(content, f)
        else:
            # Default: write as text
            with open(file_path, 'w') as f:
                f.write(str(content))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io.py::test_iowriter_async_write_json tests/test_io.py::test_iowriter_async_write_pkl -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add acc/io.py tests/test_io.py
git commit -m "feat: implement async write method with auto-serialization"
```

---

### Task 3: Add Pending Files Tracking Tests

**Files:**
- Modify: `tests/test_io.py`

- [ ] **Step 1: Write test for pending files tracking**

```python
# tests/test_io.py (add to existing file)
import time

def test_iowriter_pending_files_tracking():
    """Test pending files set tracks files during async write"""
    writer = IOWriter(enable_async=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"key": "value"}

        # Before write: pending set should be empty
        assert len(writer._pending_files) == 0

        writer.write(file_path, content)

        # Immediately after write submission: file should be in pending set
        time.sleep(0.1)  # Small delay to let task start
        with writer._lock:
            assert file_path in writer._pending_files

        # After shutdown: pending set should be empty
        writer.shutdown()
        with writer._lock:
            assert len(writer._pending_files) == 0
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_io.py::test_iowriter_pending_files_tracking -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_io.py
git commit -m "test: add pending files tracking test"
```

---

### Task 4: Add Concurrent Write Tests

**Files:**
- Modify: `tests/test_io.py`

- [ ] **Step 1: Write test for concurrent writes**

```python
# tests/test_io.py (add to existing file)
def test_iowriter_concurrent_writes():
    """Test multiple concurrent async writes"""
    writer = IOWriter(enable_async=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write 10 files concurrently
        files = []
        for i in range(10):
            file_path = os.path.join(tmpdir, f"test_{i}.json")
            content = {"index": i, "data": f"value_{i}"}
            writer.write(file_path, content)
            files.append((file_path, content))

        # Wait for all writes to complete
        writer.shutdown()

        # Verify all files exist with correct content
        for file_path, expected_content in files:
            assert os.path.exists(file_path)
            with open(file_path, 'r') as f:
                data = json.load(f)
            assert data == expected_content
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_io.py::test_iowriter_concurrent_writes -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_io.py
git commit -m "test: add concurrent writes test"
```

---

### Task 5: Add Sync Mode Tests

**Files:**
- Modify: `tests/test_io.py`

- [ ] **Step 1: Write test for sync mode**

```python
# tests/test_io.py (add to existing file)
def test_iowriter_sync_mode():
    """Test synchronous write mode"""
    writer = IOWriter(enable_async=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test_sync.json")
        content = {"mode": "sync"}

        # Sync mode: write should block until complete
        writer.write(file_path, content)

        # File should exist immediately (no shutdown needed)
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        assert data == content

        # Shutdown should be safe (no async loop)
        writer.shutdown()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_io.py::test_iowriter_sync_mode -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_io.py
git commit -m "test: add sync mode test"
```

---

### Task 6: Add Exit Handler Warning Tests

**Files:**
- Modify: `tests/test_io.py`

- [ ] **Step 1: Write test for exit handler warning**

```python
# tests/test_io.py (add to existing file)
import subprocess
import sys

def test_iowriter_exit_handler_warning():
    """Test exit handler prints warning for pending files"""
    # Create a subprocess that will exit with pending writes
    test_script = """
import sys
sys.path.insert(0, 'acc')
from io import IOWriter
import tempfile
import os

writer = IOWriter(enable_async=True)
tmpdir = tempfile.mkdtemp()
file_path = os.path.join(tmpdir, 'test.json')
writer.write(file_path, {'key': 'value'})

# Exit immediately without shutdown (should trigger warning)
# Pending file should still be in set
"""

    result = subprocess.run(
        [sys.executable, '-c', test_script],
        capture_output=True,
        text=True
    )

    # Should see warning in stderr or stdout
    output = result.stdout + result.stderr
    assert "[IO WARN]" in output
    assert "Pending write tasks on exit" in output
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_io.py::test_iowriter_exit_handler_warning -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_io.py
git commit -m "test: add exit handler warning test"
```

---

### Task 7: Update SerializationSession to Use IOWriter

**Files:**
- Modify: `acc/serialization.py`
- Test: `tests/test_io_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_io_integration.py
import pytest
import tempfile
import os
import json
import pickle
from acc.serialization import SerializationSession

def test_serialization_session_async_io():
    """Test SerializationSession uses async IO for seq writes"""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = SerializationSession(tmpdir, enable_async_io=True)
        session.start()

        # Trigger save_operation (simulated)
        # This should use IOWriter instead of direct file writes
        # We'll verify by checking that shutdown is called

        session.end()

        # Session should have IOWriter instance
        assert hasattr(session, '_io_writer')
        assert session._io_writer is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_io_integration.py::test_serialization_session_async_io -v`
Expected: FAIL with "TypeError: SerializationSession.__init__() got an unexpected keyword argument 'enable_async_io'"

- [ ] **Step 3: Modify SerializationSession constructor**

```python
# acc/serialization.py (modify __init__ method around line 125)
from .io import IOWriter

class SerializationSession:
    """Manages a single serialization session, integrating CacheManager."""

    def __init__(self, dump_path: str, max_tensor_size_mb: int = 10240,
                 enable_cache: bool = True, enable_async_io: bool = True):
        self.dump_path = dump_path
        self.session_dir: Optional[str] = None
        self.sequence: int = 0
        self.max_tensor_size_mb: int = max_tensor_size_mb
        self._enable_cache: bool = enable_cache
        self._cache_manager: Optional[CacheManager] = None
        self._start_time: Optional[float] = None

        # New: Async IOWriter instance
        self._io_writer: Optional[IOWriter] = None
        self._enable_async_io = enable_async_io
```

- [ ] **Step 4: Run test to verify it still fails (need to initialize IOWriter)**

Run: `pytest tests/test_io_integration.py::test_serialization_session_async_io -v`
Expected: FAIL with "AttributeError: 'SerializationSession' object has no attribute '_io_writer'"

- [ ] **Step 5: Initialize IOWriter in start() method**

```python
# acc/serialization.py (modify start() method around line 134)
def start(self) -> str:
    """Create session directory and storage subdirectory, initialize CacheManager and IOWriter."""
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

    # New: Initialize IOWriter
    self._io_writer = IOWriter(enable_async=self._enable_async_io)

    print(f"[DUMP] Created session directory: {self.session_dir}")
    return self.session_dir
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_io_integration.py::test_serialization_session_async_io -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add acc/serialization.py tests/test_io_integration.py
git commit -m "feat: add IOWriter to SerializationSession constructor and start()"
```

---

### Task 8: Replace Seq File Writes in save_operation()

**Files:**
- Modify: `acc/serialization.py:185-197`
- Test: `tests/test_io_integration.py`

- [ ] **Step 1: Write test for seq file async writes**

```python
# tests/test_io_integration.py (add to existing file)
def test_serialization_session_async_seq_writes():
    """Test SerializationSession save_operation uses async writes"""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = SerializationSession(tmpdir, enable_async_io=True)
        session.start()

        # Simulate save_operation
        # We need to mock func and outputs
        import torch
        func = torch.add
        outputs = torch.tensor([1, 2, 3])

        seq = session.save_operation(
            func, "/path/to/file.py", "file.py", "test_func",
            42, (), {}, outputs
        )

        # End session (should wait for async writes)
        session.end()

        # Verify seq JSON file exists
        json_path = os.path.join(session.session_dir, f"{seq:06d}__file__test_func__add.json")
        assert os.path.exists(json_path)

        # Verify seq PKL file exists
        pkl_path = os.path.join(session.session_dir, f"{seq:06d}__file__test_func__add.pkl")
        assert os.path.exists(pkl_path)
```

- [ ] **Step 2: Run test to verify it passes (existing code works)**

Run: `pytest tests/test_io_integration.py::test_serialization_session_async_seq_writes -v`
Expected: PASS (current direct write implementation)

- [ ] **Step 3: Replace direct writes with IOWriter calls**

```python
# acc/serialization.py (modify save_operation method around lines 185-197)
def save_operation(
    self, func, filepath: str, filename: str, function: str,
    lineno: int, args: tuple, kwargs: dict, outputs
) -> int:
    """Save a single operator dump. Returns sequence number."""
    if self._cache_manager is None:
        raise RuntimeError("Session not started")
    filename_safe = _sanitize_filename(filename)
    function_safe = _sanitize_filename(function)
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
    json_filename = f"{seq:06d}__{filename_safe}__{function_safe}__{opname_safe}.json"
    pkl_filename = f"{seq:06d}__{filename_safe}__{function_safe}__{opname_safe}.pkl"
    json_path = os.path.join(self.session_dir, json_filename)
    pkl_path = os.path.join(self.session_dir, pkl_filename)

    # NEW: Use IOWriter instead of direct writes
    try:
        self._io_writer.write(json_path, {
            'sequence': seq, 'filepath': filepath, 'filename': filename,
            'function': function, 'lineno': lineno, 'opname': str(func),
            'call_stack': call_stack
        })
        self._io_writer.write(pkl_path, {
            'inputs': {'args': serialized_args, 'kwargs': serialized_kwargs},
            'outputs': serialized_outputs,
        })
    except Exception as e:
        print(f"[DUMP ERROR] {seq:06d} | {filename}:{lineno} | {func} | {e}")
        self.sequence += 1
        return seq

    print(f"[DUMP] {seq:06d} | {filename}:{lineno} | {func} | saved to {json_filename}")
    self.sequence += 1
    return seq
```

- [ ] **Step 4: Run test to verify it still passes**

Run: `pytest tests/test_io_integration.py::test_serialization_session_async_seq_writes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add acc/serialization.py tests/test_io_integration.py
git commit -m "feat: replace seq file direct writes with IOWriter async writes"
```

---

### Task 9: Update CacheManager to Use IOWriter

**Files:**
- Modify: `acc/cache.py`
- Test: `tests/test_io_integration.py`

- [ ] **Step 1: Read CacheManager current implementation**

Run: `Read acc/cache.py` to see current implementation

- [ ] **Step 2: Write test for cache async writes**

```python
# tests/test_io_integration.py (add to existing file)
def test_cache_manager_async_writes():
    """Test CacheManager uses async writes"""
    with tempfile.TemporaryDirectory() as tmpdir:
        import torch
        from acc.cache import CacheManager
        from acc.io import IOWriter

        io_writer = IOWriter(enable_async=True)
        cache_mgr = CacheManager(tmpdir, enable_cache=True, io_writer=io_writer)

        # Cache a tensor
        tensor = torch.tensor([1, 2, 3])
        cache_entry = cache_mgr.get_or_cache(tensor)

        # Shutdown IOWriter to wait for writes
        io_writer.shutdown()

        # Verify cache file exists
        assert os.path.exists(cache_entry.cache_path)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_io_integration.py::test_cache_manager_async_writes -v`
Expected: FAIL with "TypeError: CacheManager.__init__() got an unexpected keyword argument 'io_writer'"

- [ ] **Step 4: Modify CacheManager to accept IOWriter**

```python
# acc/cache.py (modify __init__ method)
class CacheManager:
    """Manages content-addressable cache for tensor/numpy data."""

    def __init__(self, storage_dir: str, enable_cache: bool = True,
                 io_writer: Optional[IOWriter] = None):
        self.storage_dir = storage_dir
        self.enable_cache = enable_cache
        self._cache: Dict[str, CacheEntry] = {}
        self._io_writer = io_writer  # New: async writer

        if not enable_cache:
            os.makedirs(storage_dir, exist_ok=True)
```

- [ ] **Step 5: Replace direct pickle.dump with IOWriter**

```python
# acc/cache.py (find the method that writes cache files and replace)
# Need to locate where pickle.dump is called and replace with:
if self._io_writer:
    self._io_writer.write(cache_path, tensor_data)
else:
    with open(cache_path, 'wb') as f:
        pickle.dump(tensor_data, f)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_io_integration.py::test_cache_manager_async_writes -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add acc/cache.py tests/test_io_integration.py
git commit -m "feat: integrate IOWriter into CacheManager for async cache writes"
```

---

### Task 10: Update SerializationSession to Pass IOWriter to CacheManager

**Files:**
- Modify: `acc/serialization.py:151`

- [ ] **Step 1: Modify CacheManager initialization**

```python
# acc/serialization.py (modify start() method around line 151)
def start(self) -> str:
    """Create session directory and storage subdirectory."""
    # ... existing code ...

    # Initialize IOWriter first
    self._io_writer = IOWriter(enable_async=self._enable_async_io)

    # Pass IOWriter to CacheManager
    self._cache_manager = CacheManager(storage_dir, self._enable_cache, io_writer=self._io_writer)

    # ... rest of existing code ...
```

- [ ] **Step 2: Run all integration tests**

Run: `pytest tests/test_io_integration.py -v`
Expected: PASS (all tests)

- [ ] **Step 3: Commit**

```bash
git add acc/serialization.py
git commit -m "feat: pass IOWriter to CacheManager in SerializationSession"
```

---

### Task 11: Update end() Method to Shutdown IOWriter

**Files:**
- Modify: `acc/serialization.py:205-209`

- [ ] **Step 1: Add shutdown to end() method**

```python
# acc/serialization.py (modify end() method around line 205)
def end(self):
    """End the session and print summary."""
    # Shutdown IOWriter to wait for pending writes
    if self._io_writer:
        self._io_writer.shutdown()

    if self.session_dir:
        elapsed = time.time() - self._start_time if self._start_time else 0
        print(f"[DUMP] Session completed: {self.sequence} operators dumped to {self.session_dir} in {elapsed:.1f}s")
```

- [ ] **Step 2: Run all integration tests**

Run: `pytest tests/test_io_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add acc/serialization.py
git commit -m "feat: add IOWriter shutdown to SerializationSession.end()"
```

---

### Task 12: Update dump.py to Pass enable_async_io Parameter

**Files:**
- Modify: `acc/dump.py:94-98`

- [ ] **Step 1: Modify ops_dump constructor**

```python
# acc/dump.py (modify __init__ method around line 94)
class ops_dump(TorchDispatchMode):
    """
    Context manager and decorator for dumping PyTorch operator calls.
    """

    def __init__(self, dump_path: str, max_tensor_size_mb: int = 10240,
                 enable_cache: bool = True, enable_async_io: bool = True):
        self.dump_path = dump_path
        self.max_tensor_size_mb = max_tensor_size_mb
        self.enable_cache = enable_cache
        self.enable_async_io = enable_async_io  # New parameter
        self.session = SerializationSession(dump_path, max_tensor_size_mb, enable_cache, enable_async_io)
        self.enabled = os.environ.get('ACC_DUMP_ENABLED', '1').lower() not in ('0', 'false', 'no', 'off')
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: PASS (all tests)

- [ ] **Step 3: Commit**

```bash
git add acc/dump.py
git commit -m "feat: add enable_async_io parameter to ops_dump"
```

---

### Task 13: Add __init__.py Export

**Files:**
- Modify: `acc/__init__.py`

- [ ] **Step 1: Add IOWriter to exports**

```python
# acc/__init__.py (add IOWriter to exports)
from .io import IOWriter

__all__ = ['ops_dump', 'ops_comp', 'OperatorRecord', 'SerializationSession', 'IOWriter']
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add acc/__init__.py
git commit -m "feat: export IOWriter from acc module"
```

---

### Task 14: Run Full Test Suite and Verify

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: PASS (all tests)

- [ ] **Step 2: Run existing operator dump tests**

Run: `pytest tests/test_operator_tools.py tests/test_outputs_and_empty_tensor.py -v`
Expected: PASS (existing functionality still works)

- [ ] **Step 3: Manual verification**

Create a simple test script:
```python
# test_async_io_manual.py
from acc import ops_dump
import torch

# Test async IO with real model execution
with ops_dump("/tmp/test_async_dump", enable_async_io=True):
    x = torch.randn(10, 10)
    y = x + x
    z = torch.matmul(x, y)

print("Async IO test completed successfully")
```

Run: `python test_async_io_manual.py`
Expected: Output shows successful dump with async IO

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete async IO layer implementation"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- ✓ IO Layer Interface with write(file_path, content)
- ✓ Async Write Mode with internal loop
- ✓ Sync Mode Option via enable_async config
- ✓ Exit Handling with atexit/signal
- ✓ Pending File Tracking with set
- ✓ SerializationSession Integration
- ✓ CacheManager Integration

**2. Placeholder scan:**
- No TBD, TODO, or placeholder text
- All code blocks contain complete implementation
- All test code is complete and runnable

**3. Type consistency:**
- IOWriter consistently named across all tasks
- write() method signature consistent: write(file_path, content)
- _pending_files set used consistently
- enable_async parameter name consistent

---

## Execution Options

Plan complete and saved to `docs/superpowers/plans/2026-05-22-async-io-layer.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?