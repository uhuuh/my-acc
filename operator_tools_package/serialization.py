"""
Serialization helpers for PyTorch Operator Dump Tool.

Provides functions to convert tensors to CPU and serialize values.
"""

import torch
import numpy as np
from typing import Any


def _tensor_to_cpu(obj: Any) -> Any:
    """Convert tensors to CPU, handle numpy arrays.
    
    Args:
        obj: Object to convert (tensor, numpy array, or container)
    
    Returns:
        Object with tensors moved to CPU
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_tensor_to_cpu(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: _tensor_to_cpu(v) for k, v in obj.items()}
    else:
        return obj


def _serialize_value(obj: Any) -> Any:
    """Serialize object with tensors moved to CPU.
    
    Args:
        obj: Object to serialize
    
    Returns:
        Serialized object with tensors on CPU
    """
    return _tensor_to_cpu(obj)


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename for dump file naming.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename safe for file system
    """
    return filename.replace('/', '_').replace('\\', '_').replace('.py', '')