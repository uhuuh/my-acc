"""Captured operator or module call."""

import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Record:
    """One captured call, mutated into placeholders before serialization."""

    seq_id: int = 0
    capturer_type: str = ""
    capturer_key: str = ""
    call_stack: list[dict] = field(default_factory=list)
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    outputs: list[Any] = field(default_factory=list)

    @property
    def save_id(self) -> str:
        key_safe = (
            self.capturer_key.replace(".", "_")
            .replace("::", "_")
            .replace(" ", "_")
        )
        return f"{self.seq_id:06d}__{self.capturer_type}__{key_safe}"

    def __post_init__(self):
        if self.call_stack is None or len(self.call_stack) == 0:
            self.call_stack = _capture_call_stack()
        if self.outputs is None:
            self.outputs = []
        elif isinstance(self.outputs, (tuple, list)):
            self.outputs = list(self.outputs)
        else:
            self.outputs = [self.outputs]

    def get_info(self) -> dict:
        """Return the JSONL record info with placeholder trees."""
        return {
            "seq_id": self.seq_id,
            "save_id": self.save_id,
            "capturer_type": self.capturer_type,
            "capturer_key": self.capturer_key,
            "call_stack": self.call_stack,
            "args": self.args,
            "kwargs": self.kwargs,
            "outputs": self.outputs,
        }

def _capture_call_stack(skip: int = 2) -> list[dict]:
    """Capture the full call stack from the calling frame (no linecache)."""
    frames = []
    f = sys._getframe(skip + 1)
    while f:
        frames.append({
            "filepath": f.f_code.co_filename,
            "lineno": f.f_lineno,
            "function": f.f_code.co_name,
        })
        f = f.f_back
    frames.reverse()
    return frames
