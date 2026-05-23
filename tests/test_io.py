# tests/test_io.py
import json
import tempfile
import os
import pickle
import time
from acc.io import IOWriter


def test_iowriter_constructor_async_mode():
    """Test IOWriter initializes with async mode by default"""
    writer = IOWriter(enable_async=True)
    assert writer.enable_async == True
    assert hasattr(writer, '_pending_files')
    assert hasattr(writer, '_loop')
    assert hasattr(writer, '_thread')
    writer.wait_complete()


def test_iowriter_constructor_sync_mode():
    """Test IOWriter initializes with sync mode"""
    writer = IOWriter(enable_async=False)
    assert writer.enable_async == False
    assert hasattr(writer, '_pending_files')
    assert writer._loop is None
    assert writer._thread is None
    writer.wait_complete()


def test_iowriter_async_write_str():
    """Test async write for JSON file"""
    writer = IOWriter(enable_async=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"msg": "hello world"}

        writer.write(file_path, content)
        writer.wait_complete()

        # Verify file exists and content is correct
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        assert data == content


def test_iowriter_async_write_obj():
    """Test async write for object (pickle) file"""
    writer = IOWriter(enable_async=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.pkl")
        content = {"tensor_data": [1, 2, 3]}

        writer.write(file_path, content)
        writer.wait_complete()

        # Verify file exists and content is correct
        assert os.path.exists(file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        assert data == content


def test_iowriter_sync_write_str():
    """Test sync write for JSON file"""
    writer = IOWriter(enable_async=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"msg": "hello sync"}

        writer.write(file_path, content)

        # No need to wait - sync write completes immediately
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        assert data == content

    writer.wait_complete()


def test_iowriter_sync_write_obj():
    """Test sync write for object (pickle) file"""
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

    writer.wait_complete()


def test_iowriter_pending_files_tracking():
    """Test pending files set tracks files during async write"""
    writer = IOWriter(enable_async=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"msg": "hello"}

        # Before write: pending set should be empty
        assert len(writer._pending_files) == 0

        writer.write(file_path, content)

        # Immediately after write submission: file should be in pending set
        assert file_path in writer._pending_files

        # After wait_complete: pending set should be empty
        writer.wait_complete()
        assert len(writer._pending_files) == 0


def test_iowriter_concurrent_writes():
    """Test multiple concurrent async writes"""
    writer = IOWriter(enable_async=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write 10 files concurrently
        files = []
        for i in range(10):
            file_path = os.path.join(tmpdir, f"test_{i}.pkl")
            content = {"index": i, "data": f"value_{i}"}
            writer.write(file_path, content)
            files.append((file_path, content))

        # Wait for all writes to complete
        writer.wait_complete()

        # Verify all files exist with correct content
        for file_path, expected_content in files:
            assert os.path.exists(file_path)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            assert data == expected_content


def test_iowriter_read_json():
    """Test read for JSON file"""
    writer = IOWriter(enable_async=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"msg": "hello read"}

        writer.write(file_path, content)
        data = writer.read(file_path)
        assert data == content


def test_iowriter_read_pickle():
    """Test read for pickle file"""
    writer = IOWriter(enable_async=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.pkl")
        content = {"key": "value", "num": 42}

        writer.write(file_path, content)
        data = writer.read(file_path)
        assert data == content