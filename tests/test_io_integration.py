# tests/test_io_integration.py
import os
import tempfile
from acc import ops_dump
from acc.serialization import create_pipeline
from acc.config import config


def test_create_pipeline():
    """Test create_pipeline creates sender with running receiver subprocess."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sender = create_pipeline(tmpdir)
        seq = sender.save_operation("test_op", (1, 2, 3), {"x": 10}, 42)
        assert seq == 0
        sender.stop()

        session_dir = sender.session_dir
        files = os.listdir(session_dir)
        json_files = [f for f in files if f.endswith('.json')]
        pkl_files = [f for f in files if f.endswith('.pkl')]
        assert len(json_files) == 1
        assert len(pkl_files) == 1


def test_ops_dump_creates_files():
    """Test ops_dump creates session files via sender/receiver pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with ops_dump(tmpdir) as dumper:
            import torch
            a = torch.randn(2, 3)
            _ = a + 1

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        assert len(dump_dirs) > 0
        session_dir = os.path.join(tmpdir, dump_dirs[0])
        json_files = [f for f in os.listdir(session_dir) if f.endswith('.json')]
        assert len(json_files) > 0


def test_ops_dump_disabled_no_files():
    """Test ops_dump with dump_enabled=False creates no files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config.update(dump_enabled=False)
        with ops_dump(tmpdir) as dumper:
            import torch
            a = torch.randn(2, 3)
            _ = a + 1
        config.update(dump_enabled=True)

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        assert len(dump_dirs) == 0
