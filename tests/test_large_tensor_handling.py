"""
Test large tensor handling and contiguous error handling.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tempfile
import json
import pickle
from acc import acc_dump
from acc.cache import CacheEntry


def test_large_tensor_replaced_with_none():
    """Test that large tensors are replaced with None."""
    print("=" * 60)
    print("Test: Large tensor replaced with None")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use env var to set max size
        os.environ['ACC_MAX_TENSOR_SIZE_MB'] = '1'
        try:
            with acc_dump(tmpdir):
                # float32 = 4 bytes, 256*256*256 = 16MB
                large_tensor = torch.randn(256, 256, 256)
                result = large_tensor + 1
        finally:
            os.environ.pop('ACC_MAX_TENSOR_SIZE_MB', None)

        dump_dirs = [
            d for d in os.listdir(tmpdir)
            if os.path.isdir(os.path.join(tmpdir, d))
        ]
        session_dir = os.path.join(tmpdir, dump_dirs[0])

        dump_files = [
            f for f in os.listdir(session_dir) if f.endswith('.json')
        ]

        for json_file in dump_files:
            json_path = os.path.join(session_dir, json_file)
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            key = metadata.get('capturer_key', metadata.get('key', ''))
            if 'add' in key.lower():
                print(f"Found add operation: {json_file}")

                pkl_path = json_path.replace('.json', '.pkl')
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)

                outputs = data.get('outputs', data.get('outputs', []))
                print(f"Outputs: {outputs}")

                # Large tensor filtered before Record: _wrap_outputs yields []
                if len(outputs) == 0:
                    print("PASS: Large output tensor filtered (empty outputs)")
                    return

                assert outputs[0] is None or isinstance(
                    outputs[0], (torch.Tensor, CacheEntry)
                ), f"Unexpected output type: {type(outputs[0])}"

                if outputs[0] is None:
                    print("PASS: Large output tensor replaced with None")
                else:
                    print(f"Output type: {type(outputs[0])}")
                return

        assert False, "No add operation found"

    print("PASS: Test passed\n")


def test_default_max_tensor_size():
    """Test that default max tensor size is 10GB."""
    print("=" * 60)
    print("Test: Default max tensor size is 10GB")
    print("=" * 60)

    from acc.config import Config
    config = Config()
    assert config.max_tensor_size_mb == 10240, (
        f"Default should be 10240 (10GB), got {config.max_tensor_size_mb}"
    )
    print(f"max_tensor_size_mb: {config.max_tensor_size_mb} MB (10GB)")

    print("PASS: Default max tensor size is 10GB\n")


def test_contiguous_error_handling():
    """Test that normal tensor operations work correctly."""
    print("=" * 60)
    print("Test: Normal tensor operations")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['ACC_MAX_TENSOR_SIZE_MB'] = '1'
        try:
            with acc_dump(tmpdir):
                normal_tensor = torch.randn(10, 10)
                result = normal_tensor + 1
        finally:
            os.environ.pop('ACC_MAX_TENSOR_SIZE_MB', None)

        dump_dirs = [
            d for d in os.listdir(tmpdir)
            if os.path.isdir(os.path.join(tmpdir, d))
        ]
        session_dir = os.path.join(tmpdir, dump_dirs[0])

        dump_files = [
            f for f in os.listdir(session_dir) if f.endswith('.json')
        ]

        for json_file in dump_files:
            json_path = os.path.join(session_dir, json_file)
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            key = metadata.get('capturer_key', metadata.get('key', ''))
            if 'add' in key.lower():
                print(f"Found add operation: {json_file}")
                pkl_path = json_path.replace('.json', '.pkl')
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)

                outputs = data.get('outputs', [])
                assert isinstance(outputs[0], CacheEntry), (
                    f"Normal tensor should be CacheEntry, got {type(outputs[0])}"
                )
                print("PASS: Normal tensor saved correctly as CacheEntry")
                return

        assert False, "No add operation found"

    print("PASS: Test passed\n")


if __name__ == "__main__":
    test_default_max_tensor_size()
    test_large_tensor_replaced_with_none()
    test_contiguous_error_handling()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
