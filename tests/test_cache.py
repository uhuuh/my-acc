import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import torch
import numpy as np
from acc.cache import CacheEntry, CacheManager


def test_cache_entry():
    print("Test: CacheEntry dataclass")
    entry = CacheEntry(cache_id="abc123", type="tensor", dtype="float32", shape=[2, 3])
    assert entry.cache_id == "abc123"
    assert entry.type == "tensor"
    assert entry.shape == [2, 3]
    print("  PASS")


def test_get_or_cache_tensor():
    print("Test: get_or_cache with tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage)
        t = torch.randn(2, 3)
        result = mgr.get_or_cache(t)
        assert isinstance(result, CacheEntry)
        assert result.type == "tensor"
        assert result.shape == [2, 3]
        result2 = mgr.get_or_cache(t)
        assert isinstance(result2, CacheEntry)
        assert result2.cache_id == result.cache_id
    print("  PASS")


def test_get_or_cache_numpy():
    print("Test: get_or_cache with numpy")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage)
        a = np.random.randn(3, 4).astype(np.float32)
        result = mgr.get_or_cache(a)
        assert isinstance(result, CacheEntry)
        assert result.type == "numpy"
        result2 = mgr.get_or_cache(a)
        assert result2.cache_id == result.cache_id
    print("  PASS")


def test_get_or_cache_scalar():
    print("Test: get_or_cache with non-tensor/numpy")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage)
        assert mgr.get_or_cache(42) == 42
        assert mgr.get_or_cache(3.14) == 3.14
        assert mgr.get_or_cache("hello") == "hello"
        assert mgr.get_or_cache(None) is None
    print("  PASS")


def test_resolve_tensor():
    print("Test: resolve tensor from CacheEntry")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage)
        t = torch.tensor([1.0, 2.0, 3.0])
        entry = mgr.get_or_cache(t)
        resolved = mgr.resolve(entry)
        assert isinstance(resolved, torch.Tensor)
        assert torch.equal(resolved, t)
    print("  PASS")


def test_resolve_nested():
    print("Test: resolve nested structure with CacheEntry")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage)
        t = torch.randn(2, 2)
        entry = mgr.get_or_cache(t)
        nested = [entry, 42, "hello"]
        resolved = mgr.resolve(nested)
        assert isinstance(resolved[0], torch.Tensor)
        assert resolved[1] == 42
        assert resolved[2] == "hello"
    print("  PASS")


def test_enable_cache_false():
    print("Test: enable_cache=False returns original objects")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage, enable_cache=False)
        t = torch.randn(2, 3)
        result = mgr.get_or_cache(t)
        assert result is t
    print("  PASS")


def test_different_tensors_different_hash():
    print("Test: different tensors get different cache_ids")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage)
        t1 = torch.ones(2, 3)
        t2 = torch.zeros(2, 3)
        e1 = mgr.get_or_cache(t1)
        e2 = mgr.get_or_cache(t2)
        assert e1.cache_id != e2.cache_id
    print("  PASS")


def test_identical_tensors_same_hash():
    print("Test: identical content yields same cache_id")
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = os.path.join(tmpdir, "storage")
        os.makedirs(storage)
        mgr = CacheManager(storage)
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([1.0, 2.0])
        e1 = mgr.get_or_cache(t1)
        e2 = mgr.get_or_cache(t2)
        assert e1.cache_id == e2.cache_id
    print("  PASS")


def main():
    print("ACC - CacheManager Unit Tests\n")
    test_cache_entry()
    test_get_or_cache_tensor()
    test_get_or_cache_numpy()
    test_get_or_cache_scalar()
    test_resolve_tensor()
    test_resolve_nested()
    test_enable_cache_false()
    test_different_tensors_different_hash()
    test_identical_tensors_same_hash()
    print("\nAll cache tests passed.")


if __name__ == "__main__":
    main()