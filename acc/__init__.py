"""
PyTorch Operator Dump & Precision Comparison Tool

Usage:
    from acc import ops_dump, ops_comp
    
    # Context manager mode
    with ops_dump("/path/to/dumps") as dumper:
        model(input)
    
    # Decorator mode
    @ops_dump("/path/to/dumps")
    def run_model(input):
        return model(input)
    
    # Compare dumps
    ops_comp("/path/to/dump1", "/path/to/dump2")
"""

from .operator_dumper import ops_dump
from .compare_dumps import ops_comp

__all__ = ['ops_dump', 'ops_comp']
__version__ = '0.1.0'