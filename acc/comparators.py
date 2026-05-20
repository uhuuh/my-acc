"""
Type Comparators for PyTorch Operator Dump Tool.

Provides ElementComparator classes for comparing different types.
"""

import torch
import numpy as np
from typing import Any, Tuple, Dict
from .formatting import format_type_info, format_comparison_result


class ElementComparator:
    """Base class for element comparison."""

    def __init__(self, a: Any, b: Any):
        """Initialize with two elements to compare."""
        self.a = a
        self.b = b

    def get_type_info(self) -> Tuple[str, str]:
        """Return type info for left and right elements."""
        raise NotImplementedError

    def compare(self) -> Dict:
        """Return comparison result (no log string)."""
        raise NotImplementedError


class NoneComparator(ElementComparator):
    """Compare two None values."""

    def get_type_info(self) -> Tuple[str, str]:
        return "None", "None"

    def compare(self) -> Dict:
        exact_match = self.a is None and self.b is None
        return {
            'exact_match': exact_match
        }


class IntComparator(ElementComparator):
    """Compare two int values."""

    def get_type_info(self) -> Tuple[str, str]:
        return f"int({self.a})", f"int({self.b})"

    def compare(self) -> Dict:
        exact_match = self.a == self.b
        diff = abs(self.a - self.b) if not exact_match else 0
        return {
            'exact_match': exact_match,
            'diff': diff
        }


class FloatComparator(ElementComparator):
    """Compare two float values."""

    def get_type_info(self) -> Tuple[str, str]:
        return f"float({self.a})", f"float({self.b})"

    def compare(self) -> Dict:
        exact_match = self.a == self.b
        diff = abs(self.a - self.b) if not exact_match else 0.0
        precision_diff = not exact_match
        return {
            'exact_match': exact_match,
            'precision_diff': precision_diff,
            'diff': diff
        }


class TensorComparator(ElementComparator):
    """Compare two tensors."""

    def get_type_info(self) -> Tuple[str, str]:
        dtype_a = str(self.a.dtype)
        shape_a = list(self.a.shape)
        dtype_b = str(self.b.dtype)
        shape_b = list(self.b.shape)
        return f"tensor(dtype={dtype_a}, shape={shape_a})", f"tensor(dtype={dtype_b}, shape={shape_b})"

    def compare(self) -> Dict:
        a = torch.from_numpy(self.a) if isinstance(self.a, np.ndarray) else self.a
        b = torch.from_numpy(self.b) if isinstance(self.b, np.ndarray) else self.b

        dtype_a_original = a.dtype
        dtype_b_original = b.dtype

        dtype_match = a.dtype == b.dtype
        shape_match = a.shape == b.shape

        if a.ndim == 0:
            a = a.unsqueeze(0)
        if b.ndim == 0:
            b = b.unsqueeze(0)

        if not shape_match:
            return {
                'dtype_match': dtype_match,
                'shape_match': False,
                'dtype_original_a': str(dtype_a_original),
                'dtype_original_b': str(dtype_b_original),
                'content_skipped': True
            }

        a_float = a.float()
        b_float = b.float()

        exact_match = torch.allclose(a_float, b_float, rtol=0, atol=0)
        match_count = (a_float == b_float).sum().item()
        total_count = a_float.numel()
        match_ratio = match_count / total_count if total_count > 0 else 1.0

        diff = torch.abs(a_float - b_float)
        max_err = diff.max().item()
        min_err = diff.min().item()
        mean_err = diff.mean().item()
        mse = torch.mean((a_float - b_float) ** 2).item()

        if total_count > 0 and a_float.norm() > 0 and b_float.norm() > 0:
            cosine = torch.nn.functional.cosine_similarity(
                a_float.view(1, -1), b_float.view(1, -1)
            ).item()
        else:
            cosine = 1.0

        precision_diff = not exact_match

        return {
            'dtype_match': dtype_match,
            'shape_match': shape_match,
            'dtype_original_a': str(dtype_a_original),
            'dtype_original_b': str(dtype_b_original),
            'exact_match': exact_match,
            'precision_diff': precision_diff,
            'match_ratio': match_ratio,
            'max_err': max_err,
            'min_err': min_err,
            'mean_err': mean_err,
            'mse': mse,
            'cosine': cosine
        }


class NumpyComparator(TensorComparator):
    """Compare two numpy arrays (uses TensorComparator logic)."""

    def get_type_info(self) -> Tuple[str, str]:
        return f"numpy(dtype={self.a.dtype}, shape={list(self.a.shape)})", f"numpy(dtype={self.b.dtype}, shape={list(self.b.shape)})"


class UnsupportedComparator(ElementComparator):
    """Compare unsupported types."""

    def get_type_info(self) -> Tuple[str, str]:
        return str(self.a), str(self.b)

    def compare(self) -> Dict:
        try:
            exact_match = self.a == self.b
        except Exception:
            exact_match = False

        return {
            'exact_match': exact_match,
            'str_a': str(self.a),
            'str_b': str(self.b)
        }


class MissingInAComparator(ElementComparator):
    """Element missing in A (only B exists)."""

    def __init__(self, b: Any):
        super().__init__(None, b)
        self.b = b

    def get_type_info(self) -> Tuple[str, str]:
        return "<missing>", format_type_info(self.b)

    def compare(self) -> Dict:
        return {
            'missing_in_A': True
        }


class MissingInBComparator(ElementComparator):
    """Element missing in B (only A exists)."""

    def __init__(self, a: Any):
        super().__init__(a, None)
        self.a = a

    def get_type_info(self) -> Tuple[str, str]:
        return format_type_info(self.a), "<missing>"

    def compare(self) -> Dict:
        return {
            'missing_in_B': True
        }


def create_comparator(a: Any, b: Any) -> ElementComparator:
    """Factory method to create appropriate comparator based on types."""
    if a is None and b is None:
        return NoneComparator(a, b)
    elif isinstance(a, int) and isinstance(b, int):
        return IntComparator(a, b)
    elif isinstance(a, float) and isinstance(b, float):
        return FloatComparator(a, b)
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return TensorComparator(a, b)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return NumpyComparator(a, b)
    elif isinstance(a, (torch.Tensor, np.ndarray)) and isinstance(b, (torch.Tensor, np.ndarray)):
        return TensorComparator(a, b)
    else:
        return UnsupportedComparator(a, b)