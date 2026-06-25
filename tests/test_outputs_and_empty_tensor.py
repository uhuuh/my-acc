"""
Test outputs saving and empty tensor handling.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tempfile
import json
import pickle
from acc import acc_dump, acc_comp
from acc.cache import CacheEntry, CacheManager
from acc.io import IOWriter
from acc.memory import MemoryAllocator


def test_saves_operator_outputs():
    """Test that acc_dump saves operator outputs."""
    print("=" * 60)
    print("Test: Saving operator outputs")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        with acc_dump(tmpdir) as dumper:
            a = torch.randn(2, 3)
            b = torch.randn(2, 3)
            c = a + b  # This should save output tensor c

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

                # Check that outputs are saved
                print(f"PKL data structure: {type(data)}")
                print(f"PKL data keys/length: {len(data) if isinstance(data, (list, dict)) else 'scalar'}")

                # New structure: {'inputs': {'args': [...], 'kwargs': {...}}, 'outputs': [...]}
                if isinstance(data, dict) and 'outputs' in data:
                    outputs = data['outputs']
                    print(f"Outputs found: {outputs}")
                    assert len(outputs) > 0, "Outputs should not be empty"
                    # Outputs may be CacheEntry objects, resolve them
                    cache_dir = os.path.join(session_dir, 'cache')
                    cache_mgr = CacheManager()
                    cache_mgr.cache_dir = cache_dir
                    cache_mgr._io = IOWriter(enable_async=False)
                    cache_mgr._pool = MemoryAllocator.create("pin")
                    cache_mgr._load_cached = {}
                    cache_mgr._started = True
                    resolved_outputs = [cache_mgr.load(o) if isinstance(o, CacheEntry) else o for o in outputs]
                    assert isinstance(resolved_outputs[0], torch.Tensor), "Output should be a tensor"
                    print(f"Resolved output tensor shape: {resolved_outputs[0].shape}")
                    print("PASS: Outputs are saved correctly")
                    return

                # If we didn't find outputs in expected structure, fail the test
                assert False, f"Outputs not found in PKL data. Data structure: {data}"

    print("PASS: Test passed\n")


def test_empty_tensor_comparison():
    """Test that empty tensors are handled specially during comparison."""
    print("=" * 60)
    print("Test: Empty tensor comparison")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        dump_dir_a = os.path.join(tmpdir, "dump_a")
        dump_dir_b = os.path.join(tmpdir, "dump_b")
        os.makedirs(dump_dir_a)
        os.makedirs(dump_dir_b)

        # Create dumps with empty tensors
        with acc_dump(dump_dir_a) as dumper:
            empty_a = torch.empty(0, 3)  # Empty tensor
            result_a = empty_a + 1

        with acc_dump(dump_dir_b) as dumper:
            empty_b = torch.empty(0, 3)  # Empty tensor
            result_b = empty_b + 1

        # Get session directories
        sessions_a = [d for d in os.listdir(dump_dir_a) if os.path.isdir(os.path.join(dump_dir_a, d))]
        sessions_b = [d for d in os.listdir(dump_dir_b) if os.path.isdir(os.path.join(dump_dir_b, d))]

        session_a_path = os.path.join(dump_dir_a, sessions_a[0])
        session_b_path = os.path.join(dump_dir_b, sessions_b[0])

        print(f"\nComparing empty tensor dumps...")
        print("-" * 60)

        # Compare dumps - should handle empty tensors without error
        acc_comp(session_a_path, session_b_path)

        print("\nPASS: Empty tensor comparison handled correctly\n")


if __name__ == "__main__":
    test_saves_operator_outputs()
    test_empty_tensor_comparison()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)