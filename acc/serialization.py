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
    return filename.replace('/', '_').replace('\\', '_').replace('.py', '')


def _sanitize_opname(opname: str) -> str:
    """Sanitize opname for dump file naming."""
    return opname.replace('.', '_').replace('::', '_')


def _serialize_inputs(args, kwargs):
    """Serialize inputs to list for PKL storage."""
    data_list = []

    for arg in args:
        if isinstance(arg, torch.Tensor):
            data_list.append(arg.detach().cpu())
        else:
            data_list.append(arg)

    if kwargs:
        data_list.append(kwargs)

    return data_list