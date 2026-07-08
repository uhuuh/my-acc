"""Value comparison used by dump reports."""

from typing import Any

import numpy as np
import torch


MISSING = object()


def compare_values(a: Any, b: Any) -> tuple[str, str, str]:
    """Return left description, right description, and comparison details."""
    if a is MISSING:
        return "<missing>", type(b).__name__, "missing_in_A"
    if b is MISSING:
        return type(a).__name__, "<missing>", "missing_in_B"
    if a is None and b is None:
        return "None", "None", "exact_match=True"
    if isinstance(a, int) and isinstance(b, int):
        exact = a == b
        return f"int({a})", f"int({b})", (
            f"exact_match={exact}, diff={abs(a - b) if not exact else 0}"
        )
    if isinstance(a, float) and isinstance(b, float):
        exact = a == b
        diff = abs(a - b) if not exact else 0.0
        return (
            f"float({a})",
            f"float({b})",
            f"exact_match={exact}, diff={diff:.6e}",
        )
    if isinstance(a, (torch.Tensor, np.ndarray)) and isinstance(
        b, (torch.Tensor, np.ndarray)
    ):
        return _compare_arrays(a, b)

    try:
        exact = a == b
        if not isinstance(exact, bool):
            exact = False
    except Exception:
        exact = False
    return str(a), str(b), f"exact_match={exact}"


def _compare_arrays(a, b) -> tuple[str, str, str]:
    kind_a = "numpy" if isinstance(a, np.ndarray) else "tensor"
    kind_b = "numpy" if isinstance(b, np.ndarray) else "tensor"
    a = torch.from_numpy(a) if isinstance(a, np.ndarray) else a
    b = torch.from_numpy(b) if isinstance(b, np.ndarray) else b

    dtype_match = a.dtype == b.dtype
    shape_match = a.shape == b.shape
    left = _describe_array(kind_a, a)
    right = _describe_array(kind_b, b)
    status = (
        f"dtype={'match' if dtype_match else 'mismatch'}, "
        f"shape={'match' if shape_match else 'mismatch'}"
    )
    if not shape_match:
        return left, right, f"{status}, content_skipped"
    if a.numel() == 0:
        return left, right, f"{status}, empty_tensor_no_content"

    a_float = a.float()
    b_float = b.float()
    diff = torch.abs(a_float - b_float)
    match_ratio = (a_float == b_float).sum().item() / a_float.numel()
    if a_float.norm() > 0 and b_float.norm() > 0:
        cosine = torch.nn.functional.cosine_similarity(
            a_float.reshape(1, -1), b_float.reshape(1, -1)
        ).item()
    else:
        cosine = 1.0

    details = [
        status,
        f"exact_match={torch.equal(a_float, b_float)}",
        f"match_ratio={match_ratio:.4f}",
        f"max_err={diff.max().item():.6e}",
        f"min_err={diff.min().item():.6e}",
        f"mean_err={diff.mean().item():.6e}",
        f"mse={torch.mean(diff ** 2).item():.6e}",
        f"cosine={cosine:.6f}",
    ]
    return left, right, ", ".join(details)


def _describe_array(kind: str, value: torch.Tensor) -> str:
    if value.is_floating_point() or value.is_complex():
        has_nan = torch.isnan(value).any().item()
        has_inf = torch.isinf(value).any().item()
        has_neg_inf = torch.isneginf(value).any().item()
    else:
        has_nan = has_inf = has_neg_inf = False
    dtype = str(value.dtype)
    if kind == "numpy":
        dtype = dtype.replace("torch.", "")
    return (
        f"{kind}(dtype={dtype}, shape={list(value.shape)}, "
        f"nan={has_nan}, inf={has_inf}, neg_inf={has_neg_inf})"
    )
