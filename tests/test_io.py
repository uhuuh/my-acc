import json
import tempfile
import os
import pickle
from acc.io import IOWriter


def test_iowriter_constructor_async_mode():
    writer = IOWriter(enable_async=True)
    assert writer.enable_async == True
    assert hasattr(writer, '_pending_files')
    assert writer._thread is None
    writer.start()
    assert writer._thread is not None
    writer.stop()


def test_iowriter_constructor_sync_mode():
    writer = IOWriter(enable_async=False)
    assert writer.enable_async == False
    assert hasattr(writer, '_pending_files')
    assert writer._thread is None


def test_iowriter_async_save_str():
    writer = IOWriter(enable_async=True)
    writer.start()
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"msg": "hello world"}
        writer.save(file_path, content)
        writer.stop()
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        assert data == content


def test_iowriter_async_save_obj():
    writer = IOWriter(enable_async=True)
    writer.start()
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.pkl")
        content = {"tensor_data": [1, 2, 3]}
        writer.save(file_path, content)
        writer.stop()
        assert os.path.exists(file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        assert data == content


def test_iowriter_sync_save_str():
    writer = IOWriter(enable_async=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"msg": "hello sync"}
        writer.save(file_path, content)
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        assert data == content
    writer.stop()


def test_iowriter_sync_save_obj():
    writer = IOWriter(enable_async=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.pkl")
        content = {"sync_data": [4, 5, 6]}
        writer.save(file_path, content)
        assert os.path.exists(file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        assert data == content
    writer.stop()


def test_iowriter_pending_files_tracking():
    writer = IOWriter(enable_async=True)
    writer.start()
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"msg": "hello"}
        assert len(writer._pending_files) == 0
        writer.save(file_path, content)
        assert file_path in writer._pending_files
        writer.stop()
        assert len(writer._pending_files) == 0


def test_iowriter_concurrent_saves():
    writer = IOWriter(enable_async=True)
    writer.start()
    with tempfile.TemporaryDirectory() as tmpdir:
        files = []
        for i in range(10):
            file_path = os.path.join(tmpdir, f"test_{i}.pkl")
            content = {"index": i, "data": f"value_{i}"}
            writer.save(file_path, content)
            files.append((file_path, content))
        writer.stop()
        for file_path, expected_content in files:
            assert os.path.exists(file_path)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            assert data == expected_content


def test_iowriter_load_json():
    writer = IOWriter(enable_async=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"msg": "hello read"}
        writer.save(file_path, content)
        data = writer.load(file_path)
        assert data == content


def test_iowriter_load_pickle():
    writer = IOWriter(enable_async=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.pkl")
        content = {"key": "value", "num": 42}
        writer.save(file_path, content)
        data = writer.load(file_path)
        assert data == content
