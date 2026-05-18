"""
Test script for operator_tools_package.

Tests basic functionality of OperatorDumper and compare_operator_dumps.
"""

import torch
import tempfile
import os
from operator_tools_package import OperatorDumper, compare_operator_dumps


def test_operator_dumper_context_manager():
    """Test OperatorDumper as context manager."""
    print("=" * 60)
    print("Test 1: OperatorDumper as context manager")
    print("=" * 60)
    
    # Create temp directory for dumps
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test context manager
        with OperatorDumper(tmpdir) as dumper:
            # Create some tensors and operations
            a = torch.randn(32, 64)
            b = torch.randn(64, 128)
            c = torch.matmul(a, b)
            d = c + torch.randn(32, 128)
            e = torch.relu(d)
        
        # Check dump directory exists
        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        print(f"\nCreated dump directories: {dump_dirs}")
        
        # Check dump files
        for dump_dir in dump_dirs:
            dump_path = os.path.join(tmpdir, dump_dir)
            dump_files = [f for f in os.listdir(dump_path) if f.endswith('.pkl')]
            print(f"Dump files in {dump_dir}: {len(dump_files)} files")
            for f in dump_files[:5]:  # Show first 5 files
                print(f"  - {f}")
    
    print("\n✓ Test 1 passed\n")


def test_operator_dumper_decorator():
    """Test OperatorDumper as decorator."""
    print("=" * 60)
    print("Test 2: OperatorDumper as decorator")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define a function to decorate
        @OperatorDumper(tmpdir)
        def run_model():
            x = torch.randn(16, 32)
            y = torch.randn(32, 64)
            z = torch.matmul(x, y)
            return torch.sum(z)
        
        # Run decorated function
        result = run_model()
        print(f"\nFunction result: {result.item():.4f}")
        
        # Check dump files
        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        for dump_dir in dump_dirs:
            dump_path = os.path.join(tmpdir, dump_dir)
            dump_files = [f for f in os.listdir(dump_path) if f.endswith('.pkl')]
            print(f"Dump files: {len(dump_files)} files")
    
    print("\n✓ Test 2 passed\n")


def test_compare_operator_dumps():
    """Test compare_operator_dumps function."""
    print("=" * 60)
    print("Test 3: compare_operator_dumps")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two dump sessions with similar operations
        dump_dir_a = os.path.join(tmpdir, "dump_a")
        dump_dir_b = os.path.join(tmpdir, "dump_b")
        os.makedirs(dump_dir_a)
        os.makedirs(dump_dir_b)
        
        # First session (A)
        with OperatorDumper(dump_dir_a) as dumper:
            a1 = torch.randn(32, 64)
            b1 = torch.randn(64, 128)
            c1 = torch.matmul(a1, b1)
            d1 = torch.relu(c1)
        
        # Second session (B) - same ops but different values
        with OperatorDumper(dump_dir_b) as dumper:
            a2 = torch.randn(32, 64)
            b2 = torch.randn(64, 128)
            c2 = torch.matmul(a2, b2)
            d2 = torch.relu(c2)
        
        # Get the session directories
        sessions_a = [d for d in os.listdir(dump_dir_a) if os.path.isdir(os.path.join(dump_dir_a, d))]
        sessions_b = [d for d in os.listdir(dump_dir_b) if os.path.isdir(os.path.join(dump_dir_b, d))]
        
        session_a_path = os.path.join(dump_dir_a, sessions_a[0])
        session_b_path = os.path.join(dump_dir_b, sessions_b[0])
        
        print(f"\nComparing: {sessions_a[0]} <-> {sessions_b[0]}")
        print("-" * 60)
        
        # Compare dumps
        compare_operator_dumps(session_a_path, session_b_path)
    
    print("\n✓ Test 3 passed\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Operator Tools Package - Integration Tests")
    print("=" * 60 + "\n")
    
    # Test import first
    print("Testing import...")
    print(f"  OperatorDumper: {OperatorDumper}")
    print(f"  compare_operator_dumps: {compare_operator_dumps}")
    print("✓ Import successful\n")
    
    # Run tests
    test_operator_dumper_context_manager()
    test_operator_dumper_decorator()
    test_compare_operator_dumps()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()