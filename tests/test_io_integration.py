# tests/test_io_integration.py
import os
import tempfile
from acc import acc_dump


def test_acc_dump_creates_files():
    """Test acc_dump creates the append-only record and tensor stores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with acc_dump(dump_path=tmpdir):
            import torch
            a = torch.randn(2, 3)
            _ = a + 1

        dump_dirs = [
            d for d in os.listdir(tmpdir)
            if os.path.isdir(os.path.join(tmpdir, d))
        ]
        assert len(dump_dirs) > 0
        session_dir = os.path.join(tmpdir, dump_dirs[0])
        assert os.path.getsize(os.path.join(session_dir, "records.jsonl")) > 0
        assert os.path.getsize(
            os.path.join(session_dir, "tensor_locations.jsonl")
        ) > 0


def test_acc_dump_disabled_no_files():
    """Test acc_dump with dump_enabled=False creates no files."""
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["ACC_DUMP_ENABLED"] = "0"
        with acc_dump(dump_path=tmpdir):
            import torch
            a = torch.randn(2, 3)
            _ = a + 1
        os.environ.pop("ACC_DUMP_ENABLED")

        dump_dirs = [
            d for d in os.listdir(tmpdir)
            if os.path.isdir(os.path.join(tmpdir, d))
        ]
        assert len(dump_dirs) == 0
