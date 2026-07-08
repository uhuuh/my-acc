import json
import tempfile
import os
import pickle
from acc.io import IOManager


def test_iomanager_constructor():
    mgr = IOManager(name="test")
    assert mgr._w is not None
    assert mgr._m is not None
    mgr.join()


def test_iomanager_save_json():
    mgr = IOManager(name="test")
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"msg": "hello world"}
        mgr.save(file_path, content)
        mgr.join()
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        assert data == content


def test_iomanager_save_pkl():
    mgr = IOManager(name="test")
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.pkl")
        content = {"tensor_data": [1, 2, 3]}
        mgr.save(file_path, content)
        mgr.join()
        assert os.path.exists(file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        assert data == content


def test_iomanager_concurrent_saves():
    mgr = IOManager(name="test")
    with tempfile.TemporaryDirectory() as tmpdir:
        files = []
        for i in range(10):
            file_path = os.path.join(tmpdir, f"test_{i}.pkl")
            content = {"index": i, "data": f"value_{i}"}
            mgr.save(file_path, content)
            files.append((file_path, content))
        mgr.join()
        for file_path, expected_content in files:
            assert os.path.exists(file_path)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            assert data == expected_content


def test_iomanager_load_json():
    mgr = IOManager(name="test")
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.json")
        content = {"msg": "hello read"}
        with open(file_path, 'w') as f:
            json.dump(content, f)
        data = mgr.load(file_path)
        assert data == content
    mgr.join()


def test_iomanager_load_pickle():
    mgr = IOManager(name="test")
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.pkl")
        content = {"key": "value", "num": 42}
        with open(file_path, 'wb') as f:
            pickle.dump(content, f)
        data = mgr.load(file_path)
        assert data == content
    mgr.join()
