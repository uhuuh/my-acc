# tests/test_io.py
import pytest
import tempfile
import json
import os
import pickle
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

def test_iowriter_async_write_json():
    """Test async write for JSON file"""
    writer = IOWriter(enable_async=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"key": "value", "number": 42}

        writer.write(file_path, content)

        # Wait for write to complete
        writer.shutdown()

        # Verify file exists and content is correct
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        assert data == content

def test_iowriter_async_write_pkl():
    """Test async write for PKL file"""
    writer = IOWriter(enable_async=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.pkl")
        content = {"tensor_data": [1, 2, 3]}

        writer.write(file_path, content)

        # Wait for write to complete
        writer.shutdown()

        # Verify file exists and content is correct
        assert os.path.exists(file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        assert data == content

def test_iowriter_sync_write_json():
    """Test sync write for JSON file"""
    writer = IOWriter(enable_async=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"sync": True, "value": 123}

        writer.write(file_path, content)

        # No need to wait - sync write completes immediately
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        assert data == content

    writer.shutdown()

def test_iowriter_sync_write_pkl():
    """Test sync write for PKL file"""
    writer = IOWriter(enable_async=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.pkl")
        content = {"sync_data": [4, 5, 6]}

        writer.write(file_path, content)

        # Verify file exists and content is correct
        assert os.path.exists(file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        assert data == content

    writer.shutdown()