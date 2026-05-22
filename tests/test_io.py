# tests/test_io.py
import pytest
from acc.io import IOWriter

def test_iowriter_constructor_async_mode():
    """Test IOWriter initializes with async mode by default"""
    writer = IOWriter(enable_async=True)
    assert writer.enable_async == True
    assert hasattr(writer, '_pending_files')
    assert hasattr(writer, '_loop')
    assert hasattr(writer, '_thread')
    writer.shutdown()

def test_iowriter_constructor_sync_mode():
    """Test IOWriter initializes with sync mode"""
    writer = IOWriter(enable_async=False)
    assert writer.enable_async == False
    assert hasattr(writer, '_pending_files')
    assert writer._loop is None
    assert writer._thread is None
    writer.shutdown()