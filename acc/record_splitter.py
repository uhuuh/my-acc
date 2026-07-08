"""Split captured records into JSON info and shared CPU tensors."""

from typing import Any

import numpy as np
import torch
from loguru import logger

from .dump_format import PLACEHOLDER_KEY, UNSUPPORTED_PREFIX
from .shared import SharedTensorManager


class RecordSplitter:
    def __init__(self, shared_manager: SharedTensorManager):
        self._shared = shared_manager
        self._next_tensor_id = 1
        self._warned_unsupported = set()

    def split(self, record):
        tensors = []
        record.args = self._visit(record.args, "args", tensors)
        record.kwargs = self._visit(record.kwargs, "kwargs", tensors)
        record.outputs = self._visit(record.outputs, "outputs", tensors)
        return record.get_info(), tensors

    def _next_id(self) -> int:
        tensor_id = self._next_tensor_id
        self._next_tensor_id += 1
        return tensor_id

    def _visit(self, value: Any, path: str, tensors: list):
        if isinstance(value, torch.Tensor):
            return self._handle_tensor(value, tensors)
        if isinstance(value, np.ndarray):
            return self._handle_numpy(value, tensors)
        if value is None or isinstance(value, (int, float)):
            return value
        if isinstance(value, dict):
            return {
                str(k): self._visit(v, f"{path}.{k}", tensors)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                self._visit(item, f"{path}[{idx}]", tensors)
                for idx, item in enumerate(value)
            ]

        type_name = f"{type(value).__module__}.{type(value).__qualname__}"
        if type_name not in self._warned_unsupported:
            self._warned_unsupported.add(type_name)
            logger.warning(
                f"[RecordSplitter] unsupported value at {path}: {type_name}"
            )
        return f"{UNSUPPORTED_PREFIX}{type_name}"

    def _handle_tensor(self, tensor, tensors: list):
        tensor_id = self._next_id()
        shared = self._shared.copy_from_tensor(tensor)
        tensors.append((tensor_id, shared))
        return {
            PLACEHOLDER_KEY: True,
            "id": tensor_id,
            "kind": "tensor",
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
        }

    def _handle_numpy(self, array, tensors: list):
        tensor_id = self._next_id()
        shared = self._shared.copy_from_numpy(array)
        tensors.append((tensor_id, shared))
        return {
            PLACEHOLDER_KEY: True,
            "id": tensor_id,
            "kind": "numpy",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
