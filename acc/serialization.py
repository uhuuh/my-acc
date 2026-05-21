"""
Serialization helpers and data structures for PyTorch Operator Dump Tool.
"""

import torch
from dataclasses import dataclass, field
from typing import Any, List, Dict


@dataclass
class OperatorDump:
    """Data structure for a single operator dump."""
    sequence: int
    filepath: str
    filename: str
    function: str
    lineno: int
    opname: str
    call_stack: List[Dict]
    inputs: List[Any] = field(default_factory=list)
    outputs: List[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> 'OperatorDump':
        """Create OperatorDump from dictionary."""
        return cls(
            sequence=data['sequence'],
            filepath=data.get('filepath', ''),
            filename=data['filename'],
            function=data['function'],
            lineno=data.get('lineno', 0),
            opname=data['opname'],
            call_stack=data.get('call_stack', []),
            inputs=data.get('inputs', []),
            outputs=data.get('outputs', [])
        )

    def to_dict(self) -> Dict:
        """Convert OperatorDump to dictionary (for JSON, without inputs/outputs)."""
        return {
            'sequence': self.sequence,
            'filepath': self.filepath,
            'filename': self.filename,
            'function': self.function,
            'lineno': self.lineno,
            'opname': self.opname,
            'call_stack': self.call_stack
        }


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename for dump file naming."""
    # Handle special filenames like <string>, <stdin>, <module>
    if filename.startswith('<') and filename.endswith('>'):
        filename = filename[1:-1]  # Remove angle brackets

    # Replace invalid characters for file naming
    result = filename.replace('/', '_').replace('\\', '_').replace('.py', '')
    # Remove any remaining invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    for char in invalid_chars:
        result = result.replace(char, '_')
    return result


def _sanitize_opname(opname: str) -> str:
    """Sanitize opname for dump file naming."""
    return opname.replace('.', '_').replace('::', '_')


def _serialize_inputs(args, kwargs, max_tensor_size_mb=10240):
    """Serialize inputs to list for PKL storage."""
    data_list = []

    for arg in args:
        if isinstance(arg, torch.Tensor):
            data_list.append(_serialize_single_tensor(arg, max_tensor_size_mb))
        else:
            data_list.append(arg)

    if kwargs:
        data_list.append(kwargs)

    return data_list


def _serialize_outputs(result, max_tensor_size_mb=10240):
    """Serialize outputs to list for PKL storage."""
    outputs_list = []

    if result is None:
        return outputs_list

    if isinstance(result, torch.Tensor):
        outputs_list.append(_serialize_single_tensor(result, max_tensor_size_mb))
    elif isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, torch.Tensor):
                outputs_list.append(_serialize_single_tensor(item, max_tensor_size_mb))
            else:
                outputs_list.append(item)
    else:
        outputs_list.append(result)

    return outputs_list


def _serialize_single_tensor(tensor, max_tensor_size_mb):
    """Serialize a single tensor with size check and contiguous handling."""
    # Check tensor size
    tensor_size_bytes = tensor.numel() * tensor.element_size()
    tensor_size_mb = tensor_size_bytes / (1024 * 1024)

    if tensor_size_mb > max_tensor_size_mb:
        print(f"[DUMP WARN] Tensor size {tensor_size_mb:.2f} MB exceeds limit {max_tensor_size_mb} MB, replacing with None")
        return None

    # Try to make tensor contiguous and move to CPU
    try:
        return tensor.detach().contiguous().cpu()
    except Exception as e:
        print(f"[DUMP WARN] Failed to make tensor contiguous: {e}, replacing with None")
        return None