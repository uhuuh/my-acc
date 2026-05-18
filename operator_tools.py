"""
PyTorch Operator Dump & Precision Comparison Tool

Single-file entry point for operator debugging tools.

Usage:
    # Context manager mode
    from operator_tools import OperatorDumper, compare_operator_dumps
    
    with OperatorDumper("/path/to/dumps") as dumper:
        model(input)
    
    # Decorator mode
    @OperatorDumper("/path/to/dumps")
    def run_model(input):
        return model(input)
    
    # Compare dumps
    compare_operator_dumps("/path/to/dump1", "/path/to/dump2")
"""

from operator_dumper import OperatorDumper
from compare_dumps import compare_operator_dumps

__all__ = ['OperatorDumper', 'compare_operator_dumps']