"""
Comparison utilities for PyTorch Operator Dump Tool.

Provides LCS matching and element comparison using OOP design.
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
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
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


def format_comparison_log(result: Dict) -> str:
    """
    Format comparison result to log string.
    
    Args:
        result: Comparison result dict from compare()
    
    Returns:
        Formatted log string
    """
    if 'missing_in_A' in result:
        return "missing_in_A"
    
    if 'missing_in_B' in result:
        return "missing_in_B"
    
    if 'unsupported' in result:
        return "type_unsupported"
    
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
        def format_info(obj):
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
        
        return "<missing>", format_info(self.b)
    
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
        def format_info(obj):
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
        
        return format_info(self.a), "<missing>"
    
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