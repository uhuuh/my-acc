import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile, torch, numpy as np, json, pickle
from acc.cache import CacheEntry, CacheManager, load_info, load_data


def _make_config(tmpdir):
    from acc.config import Config
    config = Config(dump_path=tmpdir)
    return config


def _make_record(seq_id, capturer_key, args, kwargs=None, outputs=None):
    """Create a real Record for testing (picklable across mp.Queue)."""
    # Setting ACC_CAPTURER_BACKENDS first to avoid capturer intialization noise
    from acc.record import Record
    return Record(
        seq_id=seq_id,
        capturer_type="ops",
        capturer_key=capturer_key,
        args=args,
        kwargs=kwargs or {},
        outputs=outputs or [],
    )


def test_cache_entry():
    print("Test: CacheEntry dataclass")
    entry = CacheEntry(
        cache_id=1, tensor_type="tensor", dtype="float32", shape=[2, 3]
    )
    assert entry.cache_id == 1
    assert entry.tensor_type == "tensor"
    assert entry.shape == [2, 3]
    print("  PASS")


def test_save_tensor():
    print("Test: save via CacheManager subprocess")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        mgr = CacheManager(config)

        t = torch.randn(2, 3)
        record = _make_record(1, "test", [t])
        mgr.save(record)
        mgr.join()

        # Find session subdir
        session_dir = config.dump_dir
        json_files = [
            f for f in os.listdir(session_dir)
            if f.endswith(".json") and not f.startswith("cache_hashes")
        ]
        assert len(json_files) == 1
        info = load_info(os.path.join(session_dir, json_files[0]))
        assert info["capturer_key"] == "test"

        pkl_path = os.path.join(session_dir, f"{record.save_id}.pkl")
        inputs, outputs = load_data(pkl_path, session_dir)
        assert isinstance(inputs["args"][0], torch.Tensor)
        assert torch.equal(inputs["args"][0], t)
    print("  PASS")


def test_save_numpy():
    print("Test: save numpy array via CacheManager")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        mgr = CacheManager(config)

        a = np.random.randn(3, 4).astype(np.float32)
        record = _make_record(2, "test_np", [a])
        mgr.save(record)
        mgr.join()

        session_dir = config.dump_dir
        pkl_path = os.path.join(session_dir, f"{record.save_id}.pkl")
        inputs, outputs = load_data(pkl_path, session_dir)
        assert isinstance(inputs["args"][0], np.ndarray)
        assert np.equal(inputs["args"][0], a).all()
    print("  PASS")


def test_save_scalar():
    print("Test: non-tensor values pass through")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        mgr = CacheManager(config)

        record = _make_record(3, "test_scalar", [42, 3.14, "hello", None])
        mgr.save(record)
        mgr.join()

        session_dir = config.dump_dir
        pkl_path = os.path.join(session_dir, f"{record.save_id}.pkl")
        inputs, outputs = load_data(pkl_path, session_dir)
        assert inputs["args"] == [42, 3.14, "hello", None]
    print("  PASS")


def test_different_tensors_different_cache_ids():
    print("Test: different tensors get different cache_ids")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        mgr = CacheManager(config)

        t1 = torch.ones(2, 3)
        t2 = torch.zeros(2, 3)
        r1 = _make_record(1, "a", [t1])
        r2 = _make_record(2, "b", [t2])

        mgr.save(r1)
        mgr.save(r2)
        mgr.join()

        session_dir = config.dump_dir
        with open(os.path.join(session_dir, f"{r1.save_id}.pkl"), "rb") as f:
            d1 = pickle.load(f)
        with open(os.path.join(session_dir, f"{r2.save_id}.pkl"), "rb") as f:
            d2 = pickle.load(f)

        e1 = d1["args"][0]
        e2 = d2["args"][0]
        assert isinstance(e1, CacheEntry)
        assert isinstance(e2, CacheEntry)
        assert e1.cache_id != e2.cache_id
    print("  PASS")


def main():
    print("ACC - CacheManager Unit Tests\n")
    test_cache_entry()
    test_save_tensor()
    test_save_numpy()
    test_save_scalar()
    test_different_tensors_different_cache_ids()
    print("\nAll cache tests passed.")


if __name__ == "__main__":
    main()
