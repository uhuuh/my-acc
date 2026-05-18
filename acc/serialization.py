"""
Serialization helpers for PyTorch Operator Dump Tool.

Provides functions to convert tensors to CPU and serialize values.
"""

import pickle
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


def _make_pickle_safe(obj: Any) -> Any:
    """Convert unpicklable objects to safe representations.
    
    Args:
        obj: Object to make pickle-safe
    
    Returns:
        Object safe for pickling
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_make_pickle_safe(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: _make_pickle_safe(v) for k, v in obj.items()}
    else:
        try:
            pickle.dumps(obj)
            return obj
        except (TypeError, AttributeError, RuntimeError):
            return f"<unpicklable:{type(obj).__name__}>"


def _serialize_value(obj: Any) -> Any:
    """Serialize object with tensors moved to CPU and unpicklables converted.
    
    Args:
        obj: Object to serialize
    
    Returns:
        Serialized object safe for pickling
    """
    return _make_pickle_safe(obj)


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename for dump file naming.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename safe for file system
    """
    return filename.replace('/', '_').replace('\\', '_').replace('.py', '')