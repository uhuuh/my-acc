"""
Formatting helpers for PyTorch Operator Dump Tool.

Provides functions for formatting log output.
"""

import torch
import numpy as np
from typing import Any, Dict
from .serialization import OperatorRecord, _sanitize_filename, _sanitize_opname


def format_type_info(obj: Any) -> str:
    """Format object type info for display."""
    if obj is None:
        return "None"
    elif isinstance(obj, torch.Tensor):
        return f"tensor(dtype={obj.dtype}, shape={list(obj.shape)})"
    elif isinstance(obj, np.ndarray):
        return f"numpy(dtype={obj.dtype}, shape={list(obj.shape)})"
    elif isinstance(obj, int):
        return f"int({obj})"
    elif isinstance(obj, float):
        return f"float({obj})"
    else:
        type_name = type(obj).__name__
        if isinstance(obj, (list, tuple)):
            return f"{type_name}(len={len(obj)})"
        elif isinstance(obj, dict):
            return f"dict(len={len(obj)})"
        else:
            return f"{type_name}"


def format_comparison_result(result: Dict) -> str:
    """Format comparison result to log string."""
    if 'missing_in_A' in result:
        return "missing_in_A"

    if 'missing_in_B' in result:
        return "missing_in_B"

    if 'unsupported' in result:
        return "type_unsupported"

    if 'empty_tensor' in result:
        dtype_status = 'match' if result['dtype_match'] else 'mismatch'
        shape_status = 'match' if result['shape_match'] else 'mismatch'
        return f"dtype={dtype_status}, shape={shape_status}, empty_tensor_no_content"

    if 'content_skipped' in result:
        dtype_status = 'match' if result['dtype_match'] else 'mismatch'
        shape_status = 'match' if result['shape_match'] else 'mismatch'
        return f"dtype={dtype_status}, shape={shape_status}, content_skipped"

    if 'exact_match' in result and 'diff' in result:
        exact = result['exact_match']
        diff = result['diff']
        if isinstance(diff, float):
            return f"exact_match={exact}, diff={diff:.6e}"
        else:
            return f"exact_match={exact}, diff={diff}"

    if 'str_a' in result and 'str_b' in result:
        return f"exact_match={result['exact_match']}"

    if 'dtype_match' in result and 'shape_match' in result:
        parts = []
        dtype_status = 'match' if result['dtype_match'] else 'mismatch'
        shape_status = 'match' if result['shape_match'] else 'mismatch'
        parts.append(f"dtype={dtype_status}, shape={shape_status}")
        parts.append(f"exact_match={result.get('exact_match', False)}")

        if 'match_ratio' in result:
            parts.append(f"match_ratio={result['match_ratio']:.4f}")

        if 'max_err' in result:
            parts.append(f"max_err={result['max_err']:.6e}")

        if 'min_err' in result:
            parts.append(f"min_err={result['min_err']:.6e}")

        if 'mean_err' in result:
            parts.append(f"mean_err={result['mean_err']:.6e}")

        if 'mse' in result:
            parts.append(f"mse={result['mse']:.6e}")

        if 'cosine' in result:
            parts.append(f"cosine={result['cosine']:.6f}")

        return ', '.join(parts)

    if 'exact_match' in result:
        return f"exact_match={result['exact_match']}"

    return "unknown_format"


def format_display_key(dump: OperatorRecord) -> str:
    """Format display key for log output."""
    return f"{dump.filename}({dump.opname})"


def format_dump_filename(dump: OperatorRecord) -> str:
    """Format dump filename for comparison log."""
    filename_safe = _sanitize_filename(dump.filename)
    func_name = dump.function
    opname_safe = _sanitize_opname(dump.opname)
    return f"{dump.sequence:06d}__{filename_safe}__{func_name}__{opname_safe}.json"


def format_signature(dump: OperatorRecord) -> str:
    """Build signature key for LCS matching."""
    filename = dump.filename.replace('.py', '')
    return f"{filename}::{dump.opname}"


def format_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m{int(seconds%60)}s"
    else:
        return f"{int(seconds/3600)}h{int(seconds%3600/60)}m"