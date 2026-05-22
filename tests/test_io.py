# tests/test_io.py
import pytest
import tempfile
import json
import os
import pickle
import time
import subprocess
import sys
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

def test_iowriter_pending_files_tracking():
    """Test pending files set tracks files during async write"""
    writer = IOWriter(enable_async=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"key": "value"}

        # Before write: pending set should be empty
        with writer._lock:
            assert len(writer._pending_files) == 0

        writer.write(file_path, content)

        # Immediately after write submission: file should be in pending set
        # (The file is added to pending before the coroutine is scheduled)
        with writer._lock:
            assert file_path in writer._pending_files

        # After shutdown: pending set should be empty
        writer.shutdown()
        with writer._lock:
            assert len(writer._pending_files) == 0

def test_iowriter_concurrent_writes():
    """Test multiple concurrent async writes"""
    writer = IOWriter(enable_async=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write 10 files concurrently
        files = []
        for i in range(10):
            file_path = os.path.join(tmpdir, f"test_{i}.json")
            content = {"index": i, "data": f"value_{i}"}
            writer.write(file_path, content)
            files.append((file_path, content))

        # Wait for all writes to complete
        writer.shutdown()

        # Verify all files exist with correct content
        for file_path, expected_content in files:
            assert os.path.exists(file_path)
            with open(file_path, 'r') as f:
                data = json.load(f)
            assert data == expected_content

def test_iowriter_sync_mode():
    """Test synchronous write mode"""
    writer = IOWriter(enable_async=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test_sync.json")
        content = {"mode": "sync"}

        # Sync mode: write should block until complete
        writer.write(file_path, content)

        # File should exist immediately (no shutdown needed)
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        assert data == content

        # Shutdown should be safe (no async loop)
        writer.shutdown()

def test_iowriter_exit_handler_warning():
    """Test exit handler prints warning for pending files"""
    # Create a subprocess that will exit with pending writes
    # We use a slow write operation to ensure the file is still pending at exit
    test_script = """
import sys
sys.path.insert(0, '.')
from acc.io import IOWriter
import tempfile
import os
import time

# Create a custom IOWriter with a slow serialize method
class SlowIOWriter(IOWriter):
    def _serialize_and_write(self, file_path, content):
        time.sleep(0.5)  # Make write slow to ensure pending at exit
        super()._serialize_and_write(file_path, content)

writer = SlowIOWriter(enable_async=True)
tmpdir = tempfile.mkdtemp()
file_path = os.path.join(tmpdir, 'test.json')
writer.write(file_path, {'key': 'value'})

# Verify file is in pending set before exit
print(f"PENDING: {len(writer._pending_files)}", file=sys.stderr)

# Trigger exit handlers explicitly (simulate normal exit)
# atexit handlers should be called on normal exit
sys.exit(0)
"""

    result = subprocess.run(
        [sys.executable, '-c', test_script],
        capture_output=True,
        text=True,
        cwd="C:/Users/uh/study/my-acc"
    )

    # Should see warning in stderr or stdout
    output = result.stdout + result.stderr
    assert "[IO WARN]" in output
    assert "Pending write tasks on exit" in output