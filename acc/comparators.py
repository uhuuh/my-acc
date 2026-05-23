"""
Type Comparators for PyTorch Operator Dump Tool.

Provides ElementComparator classes for comparing different types.
"""

import torch
import numpy as np
from typing import Any, Tuple


class ElementComparator:
    """Base class for element comparison."""

    def __init__(self, a: Any, b: Any):
        """Initialize with two elements to compare."""
        self.a = a
        self.b = b

    def get_type_info(self) -> Tuple[str, str]:
        """Return type info for left and right elements."""
        raise NotImplementedError

    def compare(self) -> str:
        """Return comparison result as formatted log string."""
        raise NotImplementedError


class NoneComparator(ElementComparator):
    """Compare two None values."""

    def get_type_info(self) -> Tuple[str, str]:
        return "None", "None"

    def compare(self) -> str:
        exact_match = self.a is None and self.b is None
        return f"exact_match={exact_match}"


class IntComparator(ElementComparator):
    """Compare two int values."""

    def get_type_info(self) -> Tuple[str, str]:
        return f"int({self.a})", f"int({self.b})"

    def compare(self) -> str:
        exact_match = self.a == self.b
        diff = abs(self.a - self.b) if not exact_match else 0
        if isinstance(diff, float):
            return f"exact_match={exact_match}, diff={diff:.6e}"
        return f"exact_match={exact_match}, diff={diff}"


class FloatComparator(ElementComparator):
    """Compare two float values."""

    def get_type_info(self) -> Tuple[str, str]:
        return f"float({self.a})", f"float({self.b})"

    def compare(self) -> str:
        exact_match = self.a == self.b
        diff = abs(self.a - self.b) if not exact_match else 0.0
        return f"exact_match={exact_match}, diff={diff:.6e}"


class TensorComparator(ElementComparator):
    """Compare two tensors."""

    def get_type_info(self) -> Tuple[str, str]:
        dtype_a = str(self.a.dtype)
        shape_a = list(self.a.shape)
        dtype_b = str(self.b.dtype)
        shape_b = list(self.b.shape)

        a_nan = torch.isnan(self.a).any().item()
        a_inf = torch.isinf(self.a).any().item()
        a_neg_inf = a_inf and (self.a < 0).any().item()

        b_nan = torch.isnan(self.b).any().item()
        b_inf = torch.isinf(self.b).any().item()
        b_neg_inf = b_inf and (self.b < 0).any().item()

        desc_a = f"tensor(dtype={dtype_a}, shape={shape_a}, nan={a_nan}, inf={a_inf}, neg_inf={a_neg_inf})"
        desc_b = f"tensor(dtype={dtype_b}, shape={shape_b}, nan={b_nan}, inf={b_inf}, neg_inf={b_neg_inf})"
        return desc_a, desc_b

    def compare(self) -> str:
        a = torch.from_numpy(self.a) if isinstance(self.a, np.ndarray) else self.a
        b = torch.from_numpy(self.b) if isinstance(self.b, np.ndarray) else self.b

        dtype_match = a.dtype == b.dtype
        shape_match = a.shape == b.shape

        if a.ndim == 0:
            a = a.unsqueeze(0)
        if b.ndim == 0:
            b = b.unsqueeze(0)

        dtype_status = 'match' if dtype_match else 'mismatch'
        shape_status = 'match' if shape_match else 'mismatch'

        if not shape_match:
            return f"dtype={dtype_status}, shape={shape_status}, content_skipped"

        a_float = a.float()
        b_float = b.float()

        total_count = a_float.numel()
        if total_count == 0:
            return f"dtype={dtype_status}, shape={shape_status}, empty_tensor_no_content"

        exact_match = torch.allclose(a_float, b_float, rtol=0, atol=0)
        match_count = (a_float == b_float).sum().item()
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

        parts = [f"dtype={dtype_status}, shape={shape_status}", f"exact_match={exact_match}"]
        parts.append(f"match_ratio={match_ratio:.4f}")
        parts.append(f"max_err={max_err:.6e}")
        parts.append(f"min_err={min_err:.6e}")
        parts.append(f"mean_err={mean_err:.6e}")
        parts.append(f"mse={mse:.6e}")
        parts.append(f"cosine={cosine:.6f}")
        return ', '.join(parts)


class NumpyComparator(TensorComparator):
    """Compare two numpy arrays (uses TensorComparator logic)."""

    def get_type_info(self) -> Tuple[str, str]:
        dtype_a = str(self.a.dtype)
        shape_a = list(self.a.shape)
        dtype_b = str(self.b.dtype)
        shape_b = list(self.b.shape)

        a_nan = bool(np.isnan(self.a).any())
        a_inf = bool(np.isinf(self.a).any())
        a_neg_inf = a_inf and bool((self.a < 0).any())

        b_nan = bool(np.isnan(self.b).any())
        b_inf = bool(np.isinf(self.b).any())
        b_neg_inf = b_inf and bool((self.b < 0).any())

        desc_a = f"numpy(dtype={dtype_a}, shape={shape_a}, nan={a_nan}, inf={a_inf}, neg_inf={a_neg_inf})"
        desc_b = f"numpy(dtype={dtype_b}, shape={shape_b}, nan={b_nan}, inf={b_inf}, neg_inf={b_neg_inf})"
        return desc_a, desc_b


class UnsupportedComparator(ElementComparator):
    """Compare unsupported types."""

    def get_type_info(self) -> Tuple[str, str]:
        return str(self.a), str(self.b)

    def compare(self) -> str:
        try:
            exact_match = self.a == self.b
        except Exception:
            exact_match = False
        return f"exact_match={exact_match}"


class MissingInAComparator(ElementComparator):
    """Element missing in A (only B exists)."""

    def __init__(self, b: Any):
        super().__init__(None, b)
        self.b = b

    def get_type_info(self) -> Tuple[str, str]:
        return "<missing>", type(self.b).__name__

    def compare(self) -> str:
        return "missing_in_A"


class MissingInBComparator(ElementComparator):
    """Element missing in B (only A exists)."""

    def __init__(self, a: Any):
        super().__init__(a, None)
        self.a = a

    def get_type_info(self) -> Tuple[str, str]:
        return type(self.a).__name__, "<missing>"

    def compare(self) -> str:
        return "missing_in_B"


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