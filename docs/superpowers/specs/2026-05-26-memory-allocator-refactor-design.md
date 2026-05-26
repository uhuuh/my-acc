---
title: Memory Allocator Refactor
date: 2026-05-26
---

## Summary

Refactor the memory allocator system with two subclasses: `PinMemoryAllocator` (byte-based pool with pin memory) and `NativeMemoryAllocator` (no pool, no pin). Change `acquire` interface to accept a tensor and return a CPU tensor with matching shape/dtype.

## Motivation

- Byte-based pool sizing supports different dtypes efficiently
- Unified `acquire(tensor)` interface simplifies caller code
- Clear separation between pooled pin memory and simple native allocation

## Changes

### Class Renaming

| Old Name | New Name |
|----------|----------|
| `PinMemoryAllocator` (base) | `MemoryAllocator` (base) |
| `NaiveAllocator` | `NativeMemoryAllocator` |
| `AdvancedAllocator` | `PinMemoryAllocator` |

### Interface Change

**Old:**
```python
def acquire(self, size: int) -> torch.Tensor:
    # size = element count, returns empty tensor
```

**New:**
```python
def acquire(self, tensor: torch.Tensor) -> torch.Tensor:
    # Input: any tensor (GPU or CPU)
    # Output: empty CPU tensor with same shape/dtype (no data copy)
```

### PinMemoryAllocator: Byte-Based Pool

Pool block sizes use power-of-2 **byte count** instead of element count:

```python
# Calculate bytes needed
bytes_needed = tensor.numel() * tensor.element_size()
aligned_bytes = next_pow2(bytes_needed)

# Allocate block with correct dtype
block = torch.empty(aligned_bytes // tensor.element_size(),
                    dtype=tensor.dtype, pin_memory=True)

# Return view matching original shape (empty, no data copy)
return block[:tensor.numel()].reshape(tensor.shape)
```

### NativeMemoryAllocator

Simple allocation without pool or pin memory:

```python
def acquire(self, tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(tensor).cpu()
```

### Monitoring Simplification

Remove `interval={elapsed:.1f}s total={now - self._start_time:.1f}s:` from monitoring logs.

**Old:**
```python
print(f"[POOL MONITOR] interval={elapsed:.1f}s total={now - self._start_time:.1f}s: Blocks: ...")
```

**New:**
```python
print(f"[POOL MONITOR] Blocks: {free + used} ({used} in use) | Acquires: ...")
```

Both allocators use same monitoring format (NativeMemoryAllocator logs allocations without pool stats).

### Files Modified

| File | Changes |
|------|---------|
| `acc/memory.py` | Rename classes, byte-based pool, new acquire interface, simplified monitoring |
| `acc/cache.py` | Update `Storage.materialize()` to call `acquire(tensor)` and then `.copy_(tensor)` |

### Default Allocator

```python
@classmethod
def create(cls, kind: str = "native") -> 'MemoryAllocator':
    if kind == "pin":
        return PinMemoryAllocator()
    return NativeMemoryAllocator()
```

Default is `"native"` (no pool, no pin memory).

## Behavior After Change

- `MemoryAllocator.create()` → `NativeMemoryAllocator` (default)
- `MemoryAllocator.create("native")` → `NativeMemoryAllocator` (no pool, no pin)
- `MemoryAllocator.create("pin")` → `PinMemoryAllocator` (pooled, pin memory)
- `acquire(tensor)` returns empty CPU tensor with matching shape/dtype
- Caller must call `.copy_(tensor)` to copy data
- Pool efficiently handles different dtypes via byte-based sizing