# Remove io_flush_mode Config Option Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `io_flush_mode` configuration option, making IOWriter always use explicit `stop()` flush behavior.

**Architecture:** Remove the config field, delete atexit registration logic from IOWriter, simplify `stop()` to always flush, clean up test config.

**Tech Stack:** Python, PyTorch, pytest

---

## Files Modified

| File | Change | Purpose |
|------|--------|---------|
| `acc/config.py` | Delete line 15 | Remove `io_flush_mode` field |
| `acc/io.py` | Delete imports, simplify start/stop, delete method | Remove atexit logic |
| `tests/conftest.py` | Delete lines 2-5 | Remove config override |

---

### Task 1: Remove io_flush_mode from Config

**Files:**
- Modify: `acc/config.py:15`

- [ ] **Step 1: Remove the io_flush_mode field**

Delete line 15 from `acc/config.py`:

```python
# REMOVE THIS LINE:
io_flush_mode: str = "atexit"
```

The Config class should look like this after removal:

```python
@dataclass
class Config:
    dump_path: str = "."
    dump_enabled: bool = True
    max_tensor_size_mb: int = 10240
    io_monitor_interval: float = 5.0
    cache_monitor_interval: float = 5.0
    pool_monitor_interval: float = 5.0
    async_io: bool = True
    async_serialization: bool = True
    # io_flush_mode removed - always use stop mode
```

- [ ] **Step 2: Verify tests still pass**

Run: `pytest tests/ -v --tb=short`
Expected: Some tests may fail due to conftest.py still referencing io_flush_mode (we'll fix in Task 3)

- [ ] **Step 3: Commit**

```bash
git add acc/config.py
git commit -m "$(cat <<'EOF'
refactor: remove io_flush_mode from Config

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Remove atexit logic from IOWriter

**Files:**
- Modify: `acc/io.py:11, 91-92, 101-102, 105-111`

- [ ] **Step 1: Remove atexit import**

Delete line 11 from `acc/io.py`:

```python
# REMOVE THIS LINE:
import atexit
```

- [ ] **Step 2: Remove atexit registration from start()**

Edit `acc/io.py` around lines 84-92. The `start()` method should become:

```python
def start(self):
    from .config import config
    print(f"[IO] {self.name} started" + (" (async)" if self.enable_async else " (sync)"))
    if self.enable_async:
        self._last_monitor_time = time.time()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
```

Delete these lines:
```python
# REMOVE:
if config.io_flush_mode == "atexit":
    atexit.register(self._atexit_flush)
```

- [ ] **Step 3: Simplify stop() to always flush**

Edit `acc/io.py` around lines 94-103. The `stop()` method should become:

```python
def stop(self):
    if not self.enable_async:
        print(f"[IO] {self.name} stopped")
        return
    self._queue.put(None)
    self._stopped = True
    self._flush()
    print(f"[IO] {self.name} stopped")
```

Delete the conditional flush check:
```python
# REMOVE:
from .config import config
if config.io_flush_mode == "stop":
    self._flush()
```

(Always call `_flush()` now, no conditional needed)

- [ ] **Step 4: Remove _atexit_flush method**

Delete lines 105-111 from `acc/io.py`:

```python
# REMOVE THIS ENTIRE METHOD:
def _atexit_flush(self):
    if not self.enable_async:
        return
    if not self._stopped:
        self._queue.put(None)
    self._flush()
```

- [ ] **Step 5: Verify tests pass**

Run: `pytest tests/ -v --tb=short`
Expected: Tests may still fail due to conftest.py (fix in Task 3)

- [ ] **Step 6: Commit**

```bash
git add acc/io.py
git commit -m "$(cat <<'EOF'
refactor: remove atexit logic from IOWriter

Always flush on explicit stop() call, remove atexit handler.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Remove io_flush_mode from test conftest

**Files:**
- Modify: `tests/conftest.py:1-5`

- [ ] **Step 1: Remove io_flush_mode config override**

Edit `tests/conftest.py` to remove the io_flush_mode setting. The file should become:

```python
# conftest.py - no special IO config needed now
```

Or simply delete lines 2-5, keeping line 1 if needed, or make the file empty/minimal.

Actually, looking at the file:
```python
import os
os.environ["ACC_IO_FLUSH_MODE"] = "stop"

from acc.config import config
config.update(io_flush_mode="stop")
```

The entire file can be simplified or removed. If no other test setup is needed, the file can be deleted entirely. For safety, let's just remove the io_flush_mode lines:

```python
# Test configuration - no special setup needed
```

Or if the file has no other content, delete it entirely.

- [ ] **Step 2: Verify all tests pass**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "$(cat <<'EOF'
refactor: remove io_flush_mode from test conftest

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Final verification

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Verify no io_flush_mode references remain**

Run: `grep -r "io_flush_mode" acc/ tests/`
Expected: No matches found

- [ ] **Step 3: Verify no atexit import remains in io.py**

Run: `grep "import atexit" acc/io.py`
Expected: No matches found

---

## Self-Review Checklist

- [x] Spec coverage: All requirements covered (remove config field, remove atexit logic, update tests)
- [x] No placeholders: All steps have concrete code/commands
- [x] Type consistency: No type changes needed (removal only)