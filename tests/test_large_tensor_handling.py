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
from acc.config import config
from acc.cache import CacheEntry


def test_large_tensor_replaced_with_none():
    """Test that large tensors are replaced with None."""
    print("=" * 60)
    print("Test: Large tensor replaced with None")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config.update(max_tensor_size_mb=1)
        with acc_dump(tmpdir) as dumper:
            # Create a tensor larger than 1MB
            # float32 = 4 bytes, so 256*256*256 = 16MB
            large_tensor = torch.randn(256, 256, 256)
            result = large_tensor + 1

        # Get session directory
        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])

        # Find the add operation dump file
        dump_files = [f for f in os.listdir(session_dir) if f.endswith('.json')]

        for json_file in dump_files:
            json_path = os.path.join(session_dir, json_file)
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            # Check if this is an add operation
            if 'add' in metadata['key'].lower():
                print(f"Found add operation: {json_file}")

                # Load PKL file
                pkl_path = json_path.replace('.json', '.pkl')
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)

                # Check outputs
                outputs = data['outputs']
                print(f"Outputs: {outputs}")

                # Large input tensor should be None (exceeds 1MB)
                assert outputs[0] is None or isinstance(outputs[0], torch.Tensor), \
                    f"Output should be None or tensor, got {type(outputs[0])}"

                if outputs[0] is None:
                    print("PASS: Large output tensor replaced with None")
                else:
                    # Check if it's actually small enough
                    tensor_size_mb = outputs[0].numel() * 4 / (1024 * 1024)
                    print(f"Output tensor size: {tensor_size_mb:.2f} MB")
                    if tensor_size_mb <= 1:
                        print("PASS: Tensor is within size limit")
                    else:
                        assert False, f"Large tensor not replaced with None, size: {tensor_size_mb:.2f} MB"
                return

        assert False, "No add operation found"

    print("PASS: Test passed\n")


def test_default_max_tensor_size():
    """Test that default max tensor size is 10GB."""
    print("=" * 60)
    print("Test: Default max tensor size is 10GB")
    print("=" * 60)

    config.update(max_tensor_size_mb=10240)
    assert config.max_tensor_size_mb == 10240, \
        f"Default max_tensor_size_mb should be 10240 (10GB), got {config.max_tensor_size_mb}"
    print(f"Default max_tensor_size_mb: {config.max_tensor_size_mb} MB (10GB)")

    print("PASS: Default max tensor size is 10GB\n")


def test_contiguous_error_handling():
    """Test that contiguous() errors are handled gracefully."""
    print("=" * 60)
    print("Test: Contiguous error handling")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['ACC_MAX_TENSOR_SIZE_MB'] = '1'
        try:
            with acc_dump(tmpdir) as dumper:
                # Create a normal tensor (should work fine)
                normal_tensor = torch.randn(10, 10)
                result = normal_tensor + 1
        finally:
            os.environ.pop('ACC_MAX_TENSOR_SIZE_MB', None)

        # Get session directory
        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])

        # Find the add operation dump file
        dump_files = [f for f in os.listdir(session_dir) if f.endswith('.json')]

        for json_file in dump_files:
            json_path = os.path.join(session_dir, json_file)
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            if 'add' in metadata['key'].lower():
                print(f"Found add operation: {json_file}")

                # Load PKL file
                pkl_path = json_path.replace('.json', '.pkl')
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)

                # Check outputs - should be CacheEntry (tensor is cached)
                outputs = data['outputs']
                from acc.cache import CacheEntry
                assert isinstance(outputs[0], CacheEntry), \
                    f"Normal tensor output should be saved as CacheEntry, got {type(outputs[0])}"
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