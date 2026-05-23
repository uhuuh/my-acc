"""
Formatting helpers for PyTorch Operator Dump Tool.

Provides functions for formatting log output.
"""

from .serialization import OperatorRecord, _sanitize_filename, _sanitize_opname


def format_display_key(dump: OperatorRecord) -> str:
    """Format display key for log output."""
    return f"{dump.filename}({dump.opname})"


def format_dump_filename(dump: OperatorRecord) -> str:
    """Format dump filename for comparison log."""
    filename_safe = _sanitize_filename(dump.filename)
    func_name_safe = _sanitize_filename(dump.function)
    opname_safe = _sanitize_opname(dump.opname)
    return f"{dump.sequence:06d}__{filename_safe}__{func_name_safe}__{opname_safe}.json"


def format_signature(dump: OperatorRecord) -> str:
    """Build signature key for LCS matching."""
    filename = dump.filename.replace('.py', '')
    return f"{filename}::{dump.opname}"


def format_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m{int(seconds%60)}s"
    else:
        return f"{int(seconds/3600)}h{int(seconds%3600/60)}m"