"""
Comparison utilities for PyTorch Operator Dump Tool.

Provides functions for LCS matching and element comparison.
"""

import torch
import numpy as np
from typing import Any, List, Tuple, Dict


def _lcs_length(a: List[str], b: List[str]) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Find longest common subsequence and return matched pairs.
    
    Args:
        a: First sequence of signatures
        b: Second sequence of signatures
    
    Returns:
        Tuple of (lcs_length, matched_pairs as list of (idx_a, idx_b))
    """
    m, n = len(a), len(b)
    
    # Build LCS table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Backtrack to find matched pairs
    matched_pairs = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i-1] == b[j-1]:
            matched_pairs.append((i-1, j-1))
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    matched_pairs.reverse()
    return dp[m][n], matched_pairs


def _format_type_info(obj: Any) -> str:
    """Format type info for logging.
    
    Args:
        obj: Object to format
    
    Returns:
        Type info string
    """
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
            return f"{type_name}({repr(obj)})"


def _compare_tensors(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    """
    Compare two tensors and return metrics.
    
    Args:
        a: First tensor
        b: Second tensor
    
    Returns:
        Dict with comparison metrics
    """
    # Convert numpy to tensor if needed
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
    
    # Check dtype
    dtype_match = a.dtype == b.dtype
    
    # Check shape
    shape_match = a.shape == b.shape
    
    if not dtype_match or not shape_match:
        log = f"dtype={('match' if dtype_match else 'mismatch')}, shape={('match' if shape_match else 'mismatch')}, content_skipped"
        return {
            'dtype_match': dtype_match,
            'shape_match': shape_match,
            'content_skipped': True,
            'log': log
        }
    
    # Content comparison
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    
    exact_match = torch.allclose(a, b, rtol=0, atol=0)
    match_count = (a == b).sum().item()
    total_count = a.numel()
    match_ratio = match_count / total_count if total_count > 0 else 1.0
    
    diff = torch.abs(a_flat - b_flat)
    max_err = diff.max().item()
    min_err = diff.min().item()
    mean_err = diff.mean().item()
    
    # MSE
    mse = torch.mean((a_flat - b_flat) ** 2).item()
    
    # Cosine similarity
    if total_count > 0 and a_flat.norm() > 0 and b_flat.norm() > 0:
        cosine = torch.nn.functional.cosine_similarity(
            a_flat.unsqueeze(0), b_flat.unsqueeze(0)
        ).item()
    else:
        cosine = 1.0
    
    log = f"dtype=match, shape=match, exact_match={exact_match}, match_ratio={match_ratio:.4f}, max_err={max_err:.6e}, min_err={min_err:.6e}, mean_err={mean_err:.6e}, mse={mse:.6e}, cosine={cosine:.6f}"
    
    return {
        'dtype_match': dtype_match,
        'shape_match': shape_match,
        'exact_match': exact_match,
        'match_ratio': match_ratio,
        'max_err': max_err,
        'min_err': min_err,
        'mean_err': mean_err,
        'mse': mse,
        'cosine': cosine,
        'log': log
    }


def _compare_element(a: Any, b: Any) -> Tuple[str, Dict[str, bool]]:
    """
    Compare two elements and return log string and stats.
    
    Args:
        a: First element
        b: Second element
    
    Returns:
        Tuple of (log string, stats dict)
    """
    a_type = _format_type_info(a)
    b_type = _format_type_info(b)
    
    stats = {
        'exact_match': False,
        'precision_diff': False,
        'dtype_mismatch': False,
        'shape_mismatch': False
    }
    
    # Both None
    if a is None and b is None:
        stats['exact_match'] = True
        return f"{a_type} | {b_type} | exact_match=True", stats
    
    # Type mismatch (one None, other not)
    if a is None or b is None:
        stats['dtype_mismatch'] = True
        return f"{a_type} | {b_type} | type_mismatch", stats
    
    # int comparison
    if isinstance(a, int) and isinstance(b, int):
        exact = a == b
        diff = abs(a - b) if not exact else 0
        stats['exact_match'] = exact
        return f"{a_type} | {b_type} | exact_match={exact}, diff={diff}", stats
    
    # float comparison
    if isinstance(a, float) and isinstance(b, float):
        exact = a == b
        diff = abs(a - b) if not exact else 0.0
        stats['exact_match'] = exact
        if not exact:
            stats['precision_diff'] = True
        return f"{a_type} | {b_type} | exact_match={exact}, diff={diff:.6e}", stats
    
    # Tensor or numpy comparison
    if isinstance(a, (torch.Tensor, np.ndarray)) and isinstance(b, (torch.Tensor, np.ndarray)):
        metrics = _compare_tensors(a, b)
        
        if not metrics['dtype_match']:
            stats['dtype_mismatch'] = True
        elif not metrics['shape_match']:
            stats['shape_mismatch'] = True
        elif metrics['exact_match']:
            stats['exact_match'] = True
        else:
            stats['precision_diff'] = True
        
        return f"{a_type} | {b_type} | {metrics['log']}", stats
    
    # Unsupported types
    return f"{a_type} | {b_type} | type_unsupported", stats