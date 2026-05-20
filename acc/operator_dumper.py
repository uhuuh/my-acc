"""
Operator Dumper for PyTorch Operator Dump Tool.

Provides ops_dump class for capturing operator calls.
"""

import os
import pickle
import traceback
import uuid
from datetime import datetime
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode
from .serialization import _serialize_inputs, _sanitize_filename


class ops_dump(TorchDispatchMode):
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
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = "None"
        
        pid = os.getpid()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        session_id = uuid.uuid4().hex[:8]
        
        self.session_dir = os.path.join(
            self.dump_path,
            f"{rank}-{pid}-{timestamp}-{session_id}"
        )
        
        os.makedirs(self.session_dir, exist_ok=False)
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
        import json
        
        stack = traceback.extract_stack()
        
        filepath = ""
        filename = "<global>"
        func_name = ""
        lineno = 0
        
        for frame_info in reversed(stack):
            if not frame_info.filename.endswith('operator_dumper.py'):
                filepath = frame_info.filename
                filename = os.path.basename(frame_info.filename)
                func_name = frame_info.name
                lineno = frame_info.lineno
                break
        
        call_stack = ''.join(traceback.format_stack())
        
        filename_safe = _sanitize_filename(filename)
        opname_safe = str(func).replace('.', '_').replace('::', '_')
        
        data_list = _serialize_inputs(args, kwargs)
        
        json_data = {
            'sequence': self.sequence,
            'filepath': filepath,
            'filename': filename,
            'function': func_name,
            'lineno': lineno,
            'opname': str(func),
            'call_stack': call_stack
        }
        
        json_filename = f"{self.sequence:06d}__{filename_safe}__{func_name}__{opname_safe}.json"
        pkl_filename = f"{self.sequence:06d}__{filename_safe}__{func_name}__{opname_safe}.pkl"
        
        json_path = os.path.join(self.session_dir, json_filename)
        pkl_path = os.path.join(self.session_dir, pkl_filename)
        
        try:
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            with open(pkl_path, 'wb') as f:
                pickle.dump(data_list, f)
        except Exception as e:
            print(f"[DUMP ERROR] {self.sequence:06d} | {filename}:{lineno} | {func} | {e}")
            self.sequence += 1
            return
        
        print(f"[DUMP] {self.sequence:06d} | {filename}:{lineno} | {func} | saved to {json_filename}")
        self.sequence += 1
    
    def __call__(self, func):
        """Use as decorator."""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper