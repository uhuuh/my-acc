"""Content-addressable cache for tensor/numpy deduplication."""

import hashlib
import os
import pickle
from dataclasses import dataclass
from typing import Any, List, Set, Optional

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

    def __init__(self, storage_dir: str, enable_cache: bool = True, io_writer: Optional[Any] = None):
        self.storage_dir = storage_dir
        self.enable_cache = enable_cache
        self._cached_ids: Set[str] = set()
        self._io_writer = io_writer

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
        """Resolve CacheEntry to tensor/numpy, recursively handling nested structures."""
        if isinstance(obj, CacheEntry):
            return self._load_from_storage(obj.cache_id)
        if isinstance(obj, list):
            return [self.resolve(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self.resolve(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return tuple(self.resolve(item) for item in obj)
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
        if self._io_writer is not None:
            self._io_writer.write(filepath, obj)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)

    def _load_from_storage(self, cache_id: str) -> Any:
        """Load tensor/numpy from storage/{cache_id}.pkl."""
        filepath = os.path.join(self.storage_dir, f"{cache_id}.pkl")
        with open(filepath, 'rb') as f:
            return pickle.load(f)