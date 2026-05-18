"""
Operator Dumper for PyTorch Operator Dump Tool.

Provides OperatorDumper class for capturing operator calls.
"""

import os
import pickle
import traceback
import inspect
from datetime import datetime
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from .serialization import _serialize_value, _sanitize_filename


class OperatorDumper(TorchDispatchMode):
    """Context manager and decorator for dumping PyTorch operator calls."""
    
    def __init__(self, dump_path: str):
        """
        Initialize OperatorDumper.
        
        Args:
            dump_path: Base path for dump output
        """
        self.dump_path = dump_path
        self.session_dir = None
        self.sequence = 0
        self._active = False
    
    def __enter__(self):
        """Enter context manager, create session directory."""
        pid = os.getpid()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.dump_path, f"{pid}_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.sequence = 0
        self._active = True
        print(f"[DUMP] Created session directory: {self.session_dir}")
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self._active = False
        print(f"[DUMP] Session completed: {self.sequence} operators dumped to {self.session_dir}")
        return super().__exit__(exc_type, exc_val, exc_tb)
    
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Intercept torch operator calls."""
        if not self._active:
            return func(*args, **(kwargs or {}))
        
        kwargs = kwargs or {}
        
        # Execute the operation
        result = func(*args, **kwargs)
        
        # Dump this operation
        self._dump_operation(func, args, kwargs, result)
        
        return result
    
    def _dump_operation(self, func, args, kwargs, result):
        """Dump a single operator call."""
        # Extract caller information
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back.f_back  # Go up to actual caller
        filename = os.path.basename(caller_frame.f_code.co_filename)
        lineno = caller_frame.f_lineno
        
        # Get operator name
        opname = str(func)
        
        # Get full call stack
        call_stack = ''.join(traceback.format_stack())
        
        # Serialize inputs and outputs
        inputs = [_serialize_value(arg) for arg in args]
        if kwargs:
            inputs.append(_serialize_value(kwargs))
        outputs = [_serialize_value(result)]
        
        # Create dump data
        dump_data = {
            'sequence': self.sequence,
            'filename': filename,
            'lineno': lineno,
            'opname': opname,
            'call_stack': call_stack,
            'inputs': inputs,
            'outputs': outputs
        }
        
        # Create dump filename
        sanitized_name = _sanitize_filename(filename)
        opname_safe = opname.replace('.', '_').replace('::', '_')
        dump_filename = f"{self.sequence:04d}_{sanitized_name}_{opname_safe}.pkl"
        dump_path = os.path.join(self.session_dir, dump_filename)
        
        # Write dump file
        with open(dump_path, 'wb') as f:
            pickle.dump(dump_data, f)
        
        # Log the dump
        print(f"[DUMP] {self.sequence:04d} | {filename}:{lineno} | {opname} | saved to {dump_filename}")
        
        self.sequence += 1
    
    def __call__(self, func):
        """Use as decorator."""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper