"""Shared-memory CPU tensor allocation."""

import threading

import numpy as np
import torch
from loguru import logger

from .tensor_utils import tensor_nbytes


class SharedTensorManager:
    """Creates CPU tensors in shared memory when the system permits it."""

    def __init__(self):
        self.used_bytes = 0
        self._lock = threading.Lock()
        self.shared_enabled = self._probe_shared_memory()

    def copy_from_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        source = tensor.detach()
        shape = list(source.shape)
        dtype = source.dtype
        nbytes = source.numel() * source.element_size()
        self._add_bytes(nbytes)
        try:
            target = torch.empty(shape, dtype=dtype, device="cpu")
            if self.shared_enabled:
                target.share_memory_()
            target.copy_(source, non_blocking=False)
            return target
        except BaseException:
            self.release_nbytes(nbytes)
            raise

    def copy_from_numpy(self, array: np.ndarray) -> torch.Tensor:
        source = torch.from_numpy(np.ascontiguousarray(array))
        nbytes = tensor_nbytes(source)
        self._add_bytes(nbytes)
        try:
            target = torch.empty(list(source.shape), dtype=source.dtype, device="cpu")
            if self.shared_enabled:
                target.share_memory_()
            target.copy_(source)
            return target
        except BaseException:
            self.release_nbytes(nbytes)
            raise

    def release(self, tensor: torch.Tensor):
        self.release_nbytes(tensor_nbytes(tensor))

    def release_nbytes(self, nbytes: int):
        with self._lock:
            self.used_bytes = max(0, self.used_bytes - int(nbytes))

    def _add_bytes(self, nbytes: int):
        with self._lock:
            self.used_bytes += nbytes

    def _probe_shared_memory(self) -> bool:
        try:
            torch.empty(1, dtype=torch.uint8).share_memory_()
            return True
        except Exception as exc:
            logger.warning(
                "[SharedTensorManager] shared-memory tensor allocation failed; "
                f"falling back to in-process CPU tensors and hash threads: {exc}"
            )
            return False
