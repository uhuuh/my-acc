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
        """Return comparison result."""
        raise NotImplementedError


class NoneComparator(ElementComparator):
    """Compare two None values."""
    
    def get_type_info(self) -> Tuple[str, str]:
        return "None", "None"
    
    def compare(self) -> Dict:
        exact_match = self.a is None and self.b is None
        return {
            'exact_match': exact_match,
            'log': f"exact_match={exact_match}"
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
            'diff': diff,
            'log': f"exact_match={exact_match}, diff={diff}"
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
            'diff': diff,
            'log': f"exact_match={exact_match}, diff={diff:.6e}"
        }


class TensorComparator(ElementComparator):
    """Compare two tensors."""
    
    def get_type_info(self) -> Tuple[str, str]:
        a_dtype = self.a.dtype if isinstance(self.a, torch.Tensor) else self.a.dtype
        a_shape = list(self.a.shape)
        b_dtype = self.b.dtype if isinstance(self.b, torch.Tensor) else self.b.dtype
        b_shape = list(self.b.shape)
        return f"tensor(dtype={a_dtype}, shape={a_shape})", f"tensor(dtype={b_dtype}, shape={b_shape})"
    
    def compare(self) -> Dict:
        # Convert numpy to tensor if needed
        a = torch.from_numpy(self.a) if isinstance(self.a, np.ndarray) else self.a
        b = torch.from_numpy(self.b) if isinstance(self.b, np.ndarray) else self.b
        
        dtype_match = a.dtype == b.dtype
        shape_match = a.shape == b.shape
        
        if not dtype_match or not shape_match:
            return {
                'dtype_match': dtype_match,
                'shape_match': shape_match,
                'dtype_mismatch': not dtype_match,
                'shape_mismatch': not shape_match,
                'content_skipped': True,
                'log': f"dtype={('match' if dtype_match else 'mismatch')}, shape={('match' if shape_match else 'mismatch')}, content_skipped"
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
        mse = torch.mean((a_flat - b_flat) ** 2).item()
        
        if total_count > 0 and a_flat.norm() > 0 and b_flat.norm() > 0:
            cosine = torch.nn.functional.cosine_similarity(
                a_flat.unsqueeze(0), b_flat.unsqueeze(0)
            ).item()
        else:
            cosine = 1.0
        
        precision_diff = not exact_match
        
        return {
            'dtype_match': dtype_match,
            'shape_match': shape_match,
            'exact_match': exact_match,
            'precision_diff': precision_diff,
            'match_ratio': match_ratio,
            'max_err': max_err,
            'min_err': min_err,
            'mean_err': mean_err,
            'mse': mse,
            'cosine': cosine,
            'log': f"dtype=match, shape=match, exact_match={exact_match}, match_ratio={match_ratio:.4f}, max_err={max_err:.6e}, min_err={min_err:.6e}, mean_err={mean_err:.6e}, mse={mse:.6e}, cosine={cosine:.6f}"
        }


class NumpyComparator(TensorComparator):
    """Compare two numpy arrays (uses TensorComparator logic)."""
    
    def get_type_info(self) -> Tuple[str, str]:
        return f"numpy(dtype={self.a.dtype}, shape={list(self.a.shape)})", f"numpy(dtype={self.b.dtype}, shape={list(self.b.shape)})"


class UnsupportedComparator(ElementComparator):
    """Compare unsupported types."""
    
    def get_type_info(self) -> Tuple[str, str]:
        def format_info(obj):
            type_name = type(obj).__name__
            if isinstance(obj, (list, tuple)):
                return f"{type_name}(len={len(obj)})"
            elif isinstance(obj, dict):
                return f"dict(len={len(obj)})"
            else:
                return f"{type_name}({repr(obj)[:50]})"
        
        return format_info(self.a), format_info(self.b)
    
    def compare(self) -> Dict:
        return {
            'unsupported': True,
            'log': "type_unsupported"
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
            'missing_in_A': True,
            'log': "missing_in_A"
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
            'missing_in_B': True,
            'log': "missing_in_B"
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