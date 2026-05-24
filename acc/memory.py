"""Pin memory pool with pluggable allocator and Storage helper."""

import numpy as np
import torch
from typing import Optional, Union


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


class NaiveAllocator(PinMemoryAllocator):
    """Every acquire allocates fresh pinned memory; release discards it."""

    def acquire(self, size: int) -> torch.Tensor:
        return torch.empty(size, pin_memory=True)

    def release(self, block: torch.Tensor) -> None:
        del block


class AdvancedAllocator(PinMemoryAllocator):
    """Free-list allocator with per-size buckets."""

    def __init__(self):
        self._free: dict[int, list[torch.Tensor]] = {}

    def acquire(self, size: int) -> torch.Tensor:
        sizes = sorted(k for k in self._free if k >= size)
        for bucket_size in sizes:
            blocks = self._free[bucket_size]
            block = blocks.pop()
            if not blocks:
                del self._free[bucket_size]
            block = block.reshape(-1)
            if bucket_size > size:
                leftover = block[size:]
                self._release_block(leftover.contiguous())
                block = block[:size].contiguous()
            return block
        return torch.empty(size, pin_memory=True)

    def release(self, block: torch.Tensor) -> None:
        self._release_block(block)

    def _release_block(self, block: torch.Tensor) -> None:
        flat = block.reshape(-1)
        key = flat.numel()
        if key not in self._free:
            self._free[key] = []
        self._free[key].append(flat)


class Storage:
    """Wraps a tensor/numpy array: compute cache_id on create, materialize via allocator."""

    def __init__(self, obj: Union[torch.Tensor, np.ndarray]):
        self._obj = obj
        self._materialized: Optional[torch.Tensor] = None
        self._allocator: Optional[PinMemoryAllocator] = None
        if isinstance(obj, np.ndarray):
            t = torch.from_numpy(obj).contiguous()
        else:
            t = obj.detach().contiguous()
        self.cache_id: str = f"ptr_{t.data_ptr()}_{t.numel()}_{t._version}"
        self._dtype = t.dtype
        self._shape = list(t.shape)

    def materialize(self, allocator: PinMemoryAllocator) -> torch.Tensor:
        """Acquire pinned memory, copy tensor to CPU, return storage tensor."""
        self._allocator = allocator
        if isinstance(self._obj, np.ndarray):
            cpu_tensor = torch.from_numpy(self._obj).contiguous()
        else:
            cpu_tensor = self._obj.detach().contiguous().cpu()
        flat = cpu_tensor.reshape(-1)
        storage = allocator.acquire(flat.numel())
        storage = storage.reshape(-1)
        if storage.numel() >= flat.numel():
            storage[:flat.numel()].copy_(flat)
            self._materialized = storage[:flat.numel()].reshape(cpu_tensor.shape)
        else:
            storage2 = torch.empty(flat.shape, dtype=flat.dtype, pin_memory=True)
            storage2.copy_(flat)
            allocator.release(storage)
            self._materialized = storage2.reshape(cpu_tensor.shape)
        return self._materialized

    def release(self) -> None:
        if self._materialized is not None and self._allocator is not None:
            self._allocator.release(self._materialized)
            self._materialized = None
            self._allocator = None

    def __del__(self):
        self.release()
