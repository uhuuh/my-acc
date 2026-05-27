# Memory Allocator Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor memory allocator with two subclasses (PinMemoryAllocator with byte-based pool, NativeMemoryAllocator without pool), new `acquire(tensor)` interface returning empty CPU tensor matching shape/dtype.

**Architecture:** Rename classes, change pool sizing from elements to bytes, simplify monitoring, update callers to use new interface and explicit `.copy_()`.

**Tech Stack:** Python, PyTorch, pytest

---

## Files Modified

| File | Change | Purpose |
|------|--------|---------|
| `acc/memory.py` | Rename classes, byte-based pool, new acquire interface, simplified monitoring | Core allocator refactor |
| `acc/cache.py` | Update `Storage.materialize()` to use new interface | Caller update |

---

### Task 1: Rename base class to MemoryAllocator

**Files:**
- Modify: `acc/memory.py:8-25`

- [ ] **Step 1: Rename PinMemoryAllocator base class to MemoryAllocator**

Edit `acc/memory.py` lines 8-25. Change class name and update factory method:

```python
class MemoryAllocator:
    """Base allocator: acquire(tensor)→empty CPU tensor, release(block)→void."""

    @classmethod
    def create(cls, kind: str = "native") -> 'MemoryAllocator':
        if kind == "pin":
            return PinMemoryAllocator()
        return NativeMemoryAllocator()

    def acquire(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def release(self, block: torch.Tensor) -> None:
        raise NotImplementedError

    def pool_stats(self):
        return None
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('acc/memory.py').read())"`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add acc/memory.py
git commit -m "$(cat <<'EOF'
refactor: rename PinMemoryAllocator base to MemoryAllocator

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Rename NaiveAllocator to NativeMemoryAllocator

**Files:**
- Modify: `acc/memory.py:27-35`

- [ ] **Step 1: Rename class and update acquire interface**

Edit `acc/memory.py` lines 27-35:

```python
class NativeMemoryAllocator(MemoryAllocator):
    """Simple allocation without pool or pin memory."""

    def acquire(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(tensor).cpu()

    def release(self, block: torch.Tensor) -> None:
        del block
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('acc/memory.py').read())"`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add acc/memory.py
git commit -m "$(cat <<'EOF'
refactor: rename NaiveAllocator to NativeMemoryAllocator

New acquire(tensor) interface returns empty CPU tensor.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Rename AdvancedAllocator to PinMemoryAllocator and implement byte-based pool

**Files:**
- Modify: `acc/memory.py:37-143`

- [ ] **Step 1: Rename class and update acquire interface**

Edit `acc/memory.py` starting at line 37. This is a significant refactor:

```python
class PinMemoryAllocator(MemoryAllocator):
    """Power-of-2 byte-based free-list allocator with view-based splitting."""

    def __init__(self):
        self._free: list[list[torch.Tensor]] = []
        self._allocated: dict[int, torch.Tensor] = {}
        self._pool_bytes = 0
        self._pool_bytes_delta = 0
        self._acquire_total = 0
        self._acquire_hits = 0

    def _block_bytes(self, block):
        return block.numel() * block.element_size()

    @staticmethod
    def _next_pow2(n):
        if n <= 0:
            return 1
        return 1 << (n - 1).bit_length()

    def _ensure_bucket(self, idx):
        while len(self._free) <= idx:
            self._free.append([])

    def _find_free(self, start_idx: int):
        self._ensure_bucket(start_idx)
        for i in range(start_idx, len(self._free)):
            if self._free[i]:
                return self._free[i].pop()
        return None

    def acquire(self, tensor: torch.Tensor) -> torch.Tensor:
        """Allocate CPU tensor matching input's shape/dtype, no data copy."""
        self._acquire_total += 1
        
        # Calculate bytes needed for this tensor
        bytes_needed = tensor.numel() * tensor.element_size()
        aligned_bytes = self._next_pow2(bytes_needed)
        start_idx = aligned_bytes.bit_length() - 1

        block = self._find_free(start_idx)
        if block is not None:
            self._acquire_hits += 1
            b = self._block_bytes(block)
            self._pool_bytes -= b
            self._pool_bytes_delta -= b
        else:
            # Allocate new block with same dtype as tensor
            block_elements = aligned_bytes // tensor.element_size()
            block = torch.empty(block_elements, dtype=tensor.dtype, pin_memory=True)

        self._allocated[block.untyped_storage().data_ptr()] = block
        self._check_monitor()
        
        # Return view matching original shape
        return block[:tensor.numel()].reshape(tensor.shape)

    def release(self, block: torch.Tensor) -> None:
        key = block.untyped_storage().data_ptr()
        if key in self._allocated:
            full_block = self._allocated.pop(key)
            self._release_block(full_block)
        else:
            print(f"[POOL WARN] release: block not found in _allocated (ptr=0x{key:x}), falling back")
            self._release_block(block)

    def _release_block(self, block: torch.Tensor) -> None:
        flat = block.reshape(-1)
        n = flat.numel()
        if n == 0:
            return
        b = self._block_bytes(flat)
        aligned_bytes = self._next_pow2(b)
        idx = aligned_bytes.bit_length() - 1
        self._ensure_bucket(idx)
        self._free[idx].append(flat)
        self._pool_bytes += b
        self._pool_bytes_delta += b

    def _check_monitor(self):
        from .config import config
        now = time.time()
        # Skip first call (no elapsed time yet)
        if self._acquire_total == 1:
            return
        elapsed = now - self._last_monitor_time if hasattr(self, '_last_monitor_time') else 0
        if elapsed < config.pool_monitor_interval:
            return
        free = sum(len(bucket) for bucket in self._free)
        used = len(self._allocated)
        used_bytes = sum(self._block_bytes(b) for b in self._allocated.values())
        total_bytes = self._pool_bytes + used_bytes
        ratio = (self._acquire_hits / self._acquire_total * 100) if self._acquire_total > 0 else 0
        print(f"[POOL MONITOR] Blocks: {free + used} ({used} in use) | "
              f"Acquires: {self._acquire_total} ({self._acquire_hits} hits, {ratio:.1f}%) | "
              f"Memory: {self._format_bytes(total_bytes)} ({self._format_bytes(self._pool_bytes)} free)")
        self._last_monitor_time = now

    def _format_bytes(self, n):
        units = ['B', 'KB', 'MB', 'GB']
        value = float(n)
        unit_idx = 0
        while value >= 1024 and unit_idx < len(units) - 1:
            value /= 1024
            unit_idx += 1
        return f"{value:,.2f} {units[unit_idx]}"

    def pool_stats(self):
        delta = self._pool_bytes_delta
        self._pool_bytes_delta = 0
        return (self._pool_bytes, delta)
```

- [ ] **Step 2: Remove old _start_time tracking**

The old `_start_time` and `_last_monitor_time` initialization in acquire is no longer needed. Set `_last_monitor_time` on first monitor call instead.

- [ ] **Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('acc/memory.py').read())"`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add acc/memory.py
git commit -m "$(cat <<'EOF'
refactor: rename AdvancedAllocator to PinMemoryAllocator

Byte-based pool sizing, new acquire(tensor) interface,
simplified monitoring without interval/total.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Update Storage.materialize() in cache.py

**Files:**
- Modify: `acc/cache.py:156-165`

- [ ] **Step 1: Update Storage class imports**

Update imports at top of `acc/memory.py` (line 11) and `acc/cache.py` (line 11):

In `acc/cache.py`, the import already uses `PinMemoryAllocator`. Change to use `MemoryAllocator`:

```python
from .memory import Storage, MemoryAllocator
```

- [ ] **Step 2: Update CacheManager.start() to use default allocator**

Edit `acc/cache.py` line 50:

```python
self._pool = MemoryAllocator.create("advanced")  # or keep using "advanced" -> will map to "pin"
```

Actually, per spec, `create("pin")` returns `PinMemoryAllocator`. Update:

```python
self._pool = MemoryAllocator.create("pin")
```

- [ ] **Step 3: Update Storage.materialize() to use new interface**

Edit `acc/cache.py` lines 156-165:

```python
def materialize(self, allocator: MemoryAllocator) -> torch.Tensor:
    """Acquire pinned memory matching tensor, copy data, return storage tensor."""
    if isinstance(self._obj, np.ndarray):
        flat = torch.from_numpy(self._obj).contiguous()
    else:
        flat = self._obj.detach().contiguous()
    storage = allocator.acquire(flat)
    storage.copy_(flat)
    return storage
```

- [ ] **Step 4: Verify syntax**

Run: `python -c "import ast; ast.parse(open('acc/cache.py').read())"`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add acc/cache.py
git commit -m "$(cat <<'EOF'
refactor: update Storage.materialize() for new acquire interface

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Update imports in memory.py

**Files:**
- Modify: `acc/memory.py:11`

- [ ] **Step 1: Update import statement**

The `Storage` class uses `PinMemoryAllocator` type hint. Update to use `MemoryAllocator`:

```python
def materialize(self, allocator: MemoryAllocator) -> torch.Tensor:
```

Actually, `Storage` is defined in `memory.py` and uses the local allocator type. Check if type hint needs update.

Looking at `acc/memory.py` line 156, the `materialize` method has type hint `allocator: PinMemoryAllocator`. Change to:

```python
def materialize(self, allocator: MemoryAllocator) -> torch.Tensor:
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('acc/memory.py').read())"`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add acc/memory.py
git commit -m "$(cat <<'EOF'
refactor: update Storage.materialize type hint to MemoryAllocator

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Final verification

- [ ] **Step 1: Verify no old class names remain**

Run: `grep -E "NaiveAllocator|AdvancedAllocator|class PinMemoryAllocator" acc/memory.py`
Expected: No matches (PinMemoryAllocator should only appear as subclass, not base class)

- [ ] **Step 2: Verify import chain works**

Run: `python -c "from acc.memory import MemoryAllocator, NativeMemoryAllocator, PinMemoryAllocator, Storage"`
Expected: No errors

- [ ] **Step 3: Verify factory method works**

Run: `python -c "from acc.memory import MemoryAllocator; print(MemoryAllocator.create())"`
Expected: Prints instance of NativeMemoryAllocator

---

## Self-Review Checklist

- [x] Spec coverage: All requirements covered (rename classes, byte-based pool, new interface, simplified monitoring, caller updates)
- [x] No placeholders: All steps have concrete code/commands
- [x] Type consistency: MemoryAllocator base class used consistently in type hints