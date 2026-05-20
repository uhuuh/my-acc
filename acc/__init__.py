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

from .dump import ops_dump
from .comp import ops_comp
from .serialization import OperatorDump

__all__ = ['ops_dump', 'ops_comp', 'OperatorDump']
__version__ = '0.1.0'