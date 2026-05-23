import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import torch
import numpy as np
from acc.cache import CacheEntry, CacheManager, _extract_storage, _compute_hash
from acc.io import IOWriter


def test_cache_entry():
    print("Test: CacheEntry dataclass")
    entry = CacheEntry(cache_id="abc123", type="tensor", dtype="float32", shape=[2, 3])
    assert entry.cache_id == "abc123"
    assert entry.type == "tensor"
    assert entry.shape == [2, 3]
    print("  PASS")


def test_cache_entry_from_obj():
    print("Test: CacheEntry.from_obj")
    t = torch.randn(2, 3)
    entry = CacheEntry.from_obj(t)
    assert entry.type == "tensor"
    assert entry.shape == [2, 3]
    storage = _extract_storage(t)
    assert entry.cache_id == _compute_hash(storage)
    print("  PASS")


def test_cache_entry_to_obj():
    print("Test: CacheEntry.to_obj")
    t = torch.tensor([1.0, 2.0, 3.0])
    entry = CacheEntry.from_obj(t)
    storage = _extract_storage(t)
    restored = entry.to_obj(storage)
    assert isinstance(restored, torch.Tensor)
    assert torch.equal(restored, t)
    print("  PASS")


def test_save_tensor():
    print("Test: save with tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        io_writer = IOWriter(enable_async=False)
        mgr = CacheManager(tmpdir, io_writer)
        t = torch.randn(2, 3)
        result = mgr.save(t)
        assert isinstance(result, CacheEntry)
        assert result.type == "tensor"
        assert result.shape == [2, 3]
        # Same tensor should reuse cache_id
        result2 = mgr.save(t)
        assert result2.cache_id == result.cache_id
    print("  PASS")


def test_save_numpy():
    print("Test: save with numpy")
    with tempfile.TemporaryDirectory() as tmpdir:
        io_writer = IOWriter(enable_async=False)
        mgr = CacheManager(tmpdir, io_writer)
        a = np.random.randn(3, 4).astype(np.float32)
        result = mgr.save(a)
        assert isinstance(result, CacheEntry)
        assert result.type == "numpy"
        result2 = mgr.save(a)
        assert result2.cache_id == result.cache_id
    print("  PASS")


def test_save_scalar():
    print("Test: save with non-tensor/numpy")
    with tempfile.TemporaryDirectory() as tmpdir:
        io_writer = IOWriter(enable_async=False)
        mgr = CacheManager(tmpdir, io_writer)
        assert mgr.save(42) == 42
        assert mgr.save(3.14) == 3.14
        assert mgr.save("hello") == "hello"
        assert mgr.save(None) is None
    print("  PASS")


def test_load_tensor():
    print("Test: load tensor from CacheEntry")
    with tempfile.TemporaryDirectory() as tmpdir:
        io_writer = IOWriter(enable_async=False)
        mgr = CacheManager(tmpdir, io_writer)
        t = torch.tensor([1.0, 2.0, 3.0])
        entry = mgr.save(t)
        restored = mgr.load(entry)
        assert isinstance(restored, torch.Tensor)
        assert torch.equal(restored, t)
    print("  PASS")


def test_load_numpy():
    print("Test: load numpy from CacheEntry")
    with tempfile.TemporaryDirectory() as tmpdir:
        io_writer = IOWriter(enable_async=False)
        mgr = CacheManager(tmpdir, io_writer)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        entry = mgr.save(a)
        restored = mgr.load(entry)
        assert isinstance(restored, np.ndarray)
        assert np.equal(restored, a).all()
    print("  PASS")


def test_bfloat16_tensor():
    print("Test: BFloat16 tensor save/load")
    with tempfile.TemporaryDirectory() as tmpdir:
        io_writer = IOWriter(enable_async=False)
        mgr = CacheManager(tmpdir, io_writer)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        entry = mgr.save(t)
        restored = mgr.load(entry)
        assert isinstance(restored, torch.Tensor)
        assert restored.dtype == torch.bfloat16
        assert torch.equal(restored, t)
    print("  PASS")


def test_save_load_nested():
    print("Test: save/load nested structure")
    with tempfile.TemporaryDirectory() as tmpdir:
        io_writer = IOWriter(enable_async=False)
        mgr = CacheManager(tmpdir, io_writer)
        t1 = torch.randn(2, 2)
        t2 = torch.randn(3, 3)
        data = {'tensors': [t1, t2], 'value': 42, 'name': 'test'}
        saved = mgr.save(data)
        assert isinstance(saved['tensors'][0], CacheEntry)
        assert isinstance(saved['tensors'][1], CacheEntry)
        assert saved['value'] == 42
        assert saved['name'] == 'test'
        # Load back
        loaded = mgr.load(saved)
        assert isinstance(loaded['tensors'][0], torch.Tensor)
        assert isinstance(loaded['tensors'][1], torch.Tensor)
        assert torch.equal(loaded['tensors'][0], t1)
        assert torch.equal(loaded['tensors'][1], t2)
    print("  PASS")


def test_different_tensors_different_hash():
    print("Test: different tensors get different cache_ids")
    with tempfile.TemporaryDirectory() as tmpdir:
        io_writer = IOWriter(enable_async=False)
        mgr = CacheManager(tmpdir, io_writer)
        t1 = torch.ones(2, 3)
        t2 = torch.zeros(2, 3)
        e1 = mgr.save(t1)
        e2 = mgr.save(t2)
        assert e1.cache_id != e2.cache_id
    print("  PASS")


def test_identical_tensors_same_hash():
    print("Test: identical content yields same cache_id")
    with tempfile.TemporaryDirectory() as tmpdir:
        io_writer = IOWriter(enable_async=False)
        mgr = CacheManager(tmpdir, io_writer)
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([1.0, 2.0])
        e1 = mgr.save(t1)
        e2 = mgr.save(t2)
        assert e1.cache_id == e2.cache_id
    print("  PASS")


def test_same_storage_different_shape():
    print("Test: same storage but different shape")
    with tempfile.TemporaryDirectory() as tmpdir:
        io_writer = IOWriter(enable_async=False)
        mgr = CacheManager(tmpdir, io_writer)
        t1 = torch.arange(6)  # shape [6]
        t2 = t1.reshape(2, 3)  # shape [2, 3], same storage
        e1 = mgr.save(t1)
        e2 = mgr.save(t2)
        # Same cache_id (same storage)
        assert e1.cache_id == e2.cache_id
        # Different shape
        assert e1.shape != e2.shape
        # Load and verify shapes
        r1 = mgr.load(e1)
        r2 = mgr.load(e2)
        assert list(r1.shape) == [6]
        assert list(r2.shape) == [2, 3]
    print("  PASS")


def main():
    print("ACC - CacheManager Unit Tests\n")
    test_cache_entry()
    test_cache_entry_from_obj()
    test_cache_entry_to_obj()
    test_save_tensor()
    test_save_numpy()
    test_save_scalar()
    test_load_tensor()
    test_load_numpy()
    test_bfloat16_tensor()
    test_save_load_nested()
    test_different_tensors_different_hash()
    test_identical_tensors_same_hash()
    test_same_storage_different_shape()
    print("\nAll cache tests passed.")


if __name__ == "__main__":
    main()