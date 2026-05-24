"""Content-addressable cache for tensor/numpy deduplication."""

import os
from dataclasses import dataclass
from typing import Any, List, Set, Optional, Dict

import numpy as np
import torch

from .memory import Storage, PinMemoryAllocator
from .io import IOWriter


@dataclass
class CacheEntry:
    """Metadata for a cached tensor/numpy array."""
    cache_id: str
    type: str
    dtype: str
    shape: List[int]


def _tensor_size_mb(obj) -> float:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size() / (1024 * 1024)
    return obj.size * obj.itemsize / (1024 * 1024)


class CacheManager:
    """Tracks cache IDs, transforms tensors to CacheEntry tokens, resolves back.

    Owns PinMemoryPool and cache IOWriter for .pt storage writes.
    """

    def __init__(self, cache_dir: str, cache_io: IOWriter, allocator_type: str = "advanced",
                 max_tensor_size_mb: Optional[int] = None):
        self.cache_dir = cache_dir
        self._io = cache_io
        self._save_cached: Set[str] = set()
        self._load_cached: Dict[str, torch.Tensor] = {}
        self._pool: PinMemoryAllocator = PinMemoryAllocator.create(allocator_type)
        self._max_tensor_size_mb = max_tensor_size_mb

    def save(self, data: Any) -> Any:
        """Traverse data, replace tensors/numpy with CacheEntry, write .pt on first encounter."""
        def processor(obj: Any) -> Any:
            if not isinstance(obj, (torch.Tensor, np.ndarray)):
                return obj
            if self._max_tensor_size_mb is not None:
                size_mb = _tensor_size_mb(obj)
                if size_mb > self._max_tensor_size_mb:
                    print(f"[DUMP WARN] Tensor size {size_mb:.2f} MB exceeds limit {self._max_tensor_size_mb} MB, replacing with None")
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
                self._io.write(filepath, storage_tensor)
                self._save_cached.add(cache_id)
            return entry
        return self._traverse(data, processor)

    def load(self, data: Any) -> Any:
        """Traverse data, resolve CacheEntry back to tensors/numpy, cache in memory."""
        def processor(obj: Any) -> Any:
            if not isinstance(obj, CacheEntry):
                return obj
            if obj.cache_id not in self._load_cached:
                filepath = os.path.join(self.cache_dir, f"{obj.cache_id}.pt")
                self._load_cached[obj.cache_id] = torch.load(filepath, weights_only=False)
            storage = self._load_cached[obj.cache_id]
            if obj.type == 'tensor':
                t = storage
                if list(t.shape) != obj.shape:
                    t = t.reshape(obj.shape)
                target_dtype = getattr(torch, obj.dtype.replace('torch.', ''))
                if t.dtype != target_dtype:
                    t = t.to(target_dtype)
                return t
            arr = storage.numpy()
            if list(arr.shape) != obj.shape:
                arr = arr.reshape(obj.shape)
            return arr
        return self._traverse(data, processor)

    @staticmethod
    def _traverse(data: Any, processor) -> Any:
        if isinstance(data, dict):
            return {k: CacheManager._traverse(v, processor) for k, v in data.items()}
        if isinstance(data, list):
            return [CacheManager._traverse(item, processor) for item in data]
        if isinstance(data, tuple):
            return tuple(CacheManager._traverse(item, processor) for item in data)
        if isinstance(data, (type(None), int, float, str, bool, bytes)):
            return data
        result = processor(data)
        if result is data:
            # print(f"[DUMP WARN] Unexpected type '{type(data).__name__}', replacing with None")
            return None
        return result
