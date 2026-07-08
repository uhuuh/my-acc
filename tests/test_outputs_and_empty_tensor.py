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
from acc import acc_dump, acc_comp, acc_get


def test_saves_operator_outputs():
    """Test that acc_dump saves operator outputs."""
    print("=" * 60)
    print("Test: Saving operator outputs")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        with acc_dump(tmpdir):
            a = torch.randn(2, 3)
            b = torch.randn(2, 3)
            c = a + b

        dump_dirs = [
            d for d in os.listdir(tmpdir)
            if os.path.isdir(os.path.join(tmpdir, d))
        ]
        session_dir = os.path.join(tmpdir, dump_dirs[0])

        # Use acc_get to load data
        results = acc_get(session_dir)
        assert len(results) > 0

        for r in results:
            record = r["record"]
            if 'add' in record.capturer_key.lower():
                print(f"Found add operation: {record.save_id}")
                outputs = r["outputs"]
                assert len(outputs) > 0
                assert isinstance(outputs[0], torch.Tensor)
                print(f"Resolved output tensor shape: {outputs[0].shape}")
                print("PASS: Outputs are saved correctly")
                return

        assert False, "No add operation found"

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

        with acc_dump(dump_dir_a):
            empty_a = torch.empty(0, 3)
            result_a = empty_a + 1

        with acc_dump(dump_dir_b):
            empty_b = torch.empty(0, 3)
            result_b = empty_b + 1

        sessions_a = [
            d for d in os.listdir(dump_dir_a)
            if os.path.isdir(os.path.join(dump_dir_a, d))
        ]
        sessions_b = [
            d for d in os.listdir(dump_dir_b)
            if os.path.isdir(os.path.join(dump_dir_b, d))
        ]

        session_a_path = os.path.join(dump_dir_a, sessions_a[0])
        session_b_path = os.path.join(dump_dir_b, sessions_b[0])

        print(f"\nComparing empty tensor dumps...")
        print("-" * 60)

        acc_comp(session_a_path, session_b_path)

        print("\nPASS: Empty tensor comparison handled correctly\n")


if __name__ == "__main__":
    test_saves_operator_outputs()
    test_empty_tensor_comparison()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
