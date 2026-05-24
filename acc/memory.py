"""Pin memory pool with pluggable allocator and Storage helper."""

import time
import numpy as np
import torch


class PinMemoryAllocator:
    """Base allocator: acquire(size)→tensor, release(block)→void."""

    @classmethod
    def create(cls, kind: str = "advanced") -> 'PinMemoryAllocator':
        if kind == "advanced":
            return AdvancedAllocator()
        return NaiveAllocator()

    def acquire(self, size: int) -> torch.Tensor:
        raise NotImplementedError

    def release(self, block: torch.Tensor) -> None:
        raise NotImplementedError

    def pool_stats(self):
        return None


class NaiveAllocator(PinMemoryAllocator):
    """Every acquire allocates fresh pinned memory; release discards it."""

    def acquire(self, size: int) -> torch.Tensor:
        return torch.empty(size, pin_memory=True)

    def release(self, block: torch.Tensor) -> None:
        del block


class AdvancedAllocator(PinMemoryAllocator):
    """Power-of-2 size-class free-list allocator with view-based splitting."""

    def __init__(self):
        self._free: list[list[torch.Tensor]] = []
        self._allocated: dict[int, torch.Tensor] = {}
        self._pool_bytes = 0
        self._pool_bytes_delta = 0
        self._acquire_total = 0
        self._acquire_hits = 0
        self._start_time = 0.0
        self._last_monitor_time = 0.0

    def _check_monitor(self):
        from .config import config
        now = time.time()
        elapsed = now - self._last_monitor_time
        if elapsed < config.pool_monitor_interval:
            return
        free = sum(len(bucket) for bucket in self._free)
        used = len(self._allocated)
        used_bytes = sum(self._block_bytes(b) for b in self._allocated.values())
        total_bytes = self._pool_bytes + used_bytes
        ratio = (self._acquire_hits / self._acquire_total * 100) if self._acquire_total > 0 else 0
        print(f"[POOL MONITOR] interval={elapsed:.1f}s total={now - self._start_time:.1f}s: "
              f"Blocks: {free + used} ({used} in use) | "
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

    def acquire(self, size: int) -> torch.Tensor:
        self._acquire_total += 1
        if self._start_time == 0.0:
            self._start_time = time.time()
            self._last_monitor_time = self._start_time
        aligned = self._next_pow2(size)
        start_idx = aligned.bit_length() - 1

        block = self._find_free(start_idx)
        if block is not None:
            self._acquire_hits += 1
            b = self._block_bytes(block)
            self._pool_bytes -= b
            self._pool_bytes_delta -= b
        else:
            block = torch.empty(aligned, pin_memory=True)

        self._allocated[block.untyped_storage().data_ptr()] = block
        self._check_monitor()
        return block[:size]

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
        aligned = self._next_pow2(n)
        idx = aligned.bit_length() - 1
        self._ensure_bucket(idx)
        self._free[idx].append(flat)
        b = self._block_bytes(flat)
        self._pool_bytes += b
        self._pool_bytes_delta += b

    def pool_stats(self):
        delta = self._pool_bytes_delta
        self._pool_bytes_delta = 0
        return (self._pool_bytes, delta)


class Storage:
    """Wraps a tensor/numpy array: compute cache_id on create, materialize via allocator."""

    def __init__(self, obj: torch.Tensor | np.ndarray):
        self._obj = obj
        if isinstance(obj, np.ndarray):
            t = torch.from_numpy(obj).contiguous()
        else:
            t = obj.detach().contiguous()
        self.cache_id: str = f"ptr_{t.data_ptr()}_{t.numel()}_{t._version}"

    def materialize(self, allocator: PinMemoryAllocator) -> torch.Tensor:
        """Acquire pinned memory, copy tensor to CPU, return storage tensor."""
        if isinstance(self._obj, np.ndarray):
            flat = torch.from_numpy(self._obj).contiguous()
        else:
            flat = self._obj.detach().contiguous()
        storage = allocator.acquire(flat.numel())
        storage = storage.reshape(flat.shape)
        storage.copy_(flat)
        return storage
