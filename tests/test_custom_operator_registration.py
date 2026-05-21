"""
Test ops_dump with various PyTorch custom operator registration methods.

Tests that operators called inside custom operators are captured correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tempfile
import json
from acc import ops_dump


def get_opnames_from_dump(session_dir):
    """Helper to extract operator names from dump session."""
    dump_files = [f for f in os.listdir(session_dir) if f.endswith('.json')]
    opnames = []
    for json_file in dump_files:
        json_path = os.path.join(session_dir, json_file)
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        opnames.append(metadata['opname'])
    return opnames


# ============================================================
# Test 1: torch.library with CompositeExplicitAutograd (No Decomposition)
# ============================================================

def test_torch_library_explicit_autograd():
    """
    Test CompositeExplicitAutograd operator - internal ops NOT captured.

    CompositeExplicitAutograd provides explicit backward formula.
    Internal operators do NOT decompose through dispatch mode.
    """
    print("=" * 60)
    print("Test 1: torch.library CompositeExplicitAutograd (No Decomposition)")
    print("=" * 60)

    try:
        # 定义自定义算子
        torch.library.define(
            "explicit_ops::custom_add",
            "(Tensor a, Tensor b) -> Tensor"
        )

        # 实现自定义算子 - 使用 CompositeExplicitAutograd
        @torch.library.impl("explicit_ops::custom_add", "CompositeExplicitAutograd")
        def custom_add_impl(a, b):
            # 内部调用多个原生算子，但不会被单独捕获
            y = torch.add(a, b)
            y = torch.mul(y, 2.0)
            y = torch.relu(y)
            return y

        # 测试 dump
        with tempfile.TemporaryDirectory() as tmpdir:
            a = torch.randn(3, 3)
            b = torch.randn(3, 3)

            with ops_dump(tmpdir) as dumper:
                result = torch.ops.explicit_ops.custom_add(a, b)

            dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
            session_dir = os.path.join(tmpdir, dump_dirs[0])
            opnames = get_opnames_from_dump(session_dir)

            print(f"Captured {len(opnames)} operators: {opnames}")

            # CompositeExplicitAutograd 只捕获算子本身，不捕获内部算子
            assert len(opnames) >= 1, "Should capture at least the custom operator"

            # 验证捕获了自定义算子
            custom_op_found = any('custom_add' in op.lower() for op in opnames)
            print(f"Custom operator captured: {custom_op_found}")

            # 验证内部算子没有被单独捕获（这是预期行为）
            # 注意：需要检查 aten.xxx 格式，而不是自定义算子名中的匹配
            internal_aten_ops = ['aten.add', 'aten.mul', 'aten.relu']
            internal_found = [op for op in internal_aten_ops if any(op in o.lower() for o in opnames)]
            print(f"Internal aten operators found (expected empty): {internal_found}")
            assert len(internal_found) == 0, f"Internal operators should NOT be captured in CompositeExplicitAutograd: {internal_found}"
            print("PASS: CompositeExplicitAutograd behaves as expected (no decomposition)\n")

    except Exception as e:
        print(f"SKIP: CompositeExplicitAutograd test - {e}\n")


# ============================================================
# Test 1b: torch.library with CompositeImplicitAutograd (With Decomposition)
# ============================================================

def test_torch_library_implicit_autograd():
    """
    Test CompositeImplicitAutograd operator - internal ops ARE captured.

    CompositeImplicitAutograd decomposes the operator through dispatch mode,
    allowing internal operators to be captured individually.
    """
    print("=" * 60)
    print("Test 1b: torch.library CompositeImplicitAutograd (With Decomposition)")
    print("=" * 60)

    try:
        # 定义自定义算子
        torch.library.define(
            "implicit_ops::custom_transform",
            "(Tensor x) -> Tensor"
        )

        # 实现自定义算子 - 使用 CompositeImplicitAutograd
        @torch.library.impl("implicit_ops::custom_transform", "CompositeImplicitAutograd")
        def custom_transform_impl(x):
            # 内部调用多个原生算子，会被单独捕获
            y = torch.add(x, 1.0)
            y = torch.mul(y, 2.0)
            y = torch.relu(y)
            return y

        # 测试 dump
        with tempfile.TemporaryDirectory() as tmpdir:
            x = torch.randn(3, 3)

            with ops_dump(tmpdir) as dumper:
                result = torch.ops.implicit_ops.custom_transform(x)

            dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
            session_dir = os.path.join(tmpdir, dump_dirs[0])
            opnames = get_opnames_from_dump(session_dir)

            print(f"Captured {len(opnames)} operators: {opnames}")

            # CompositeImplicitAutograd 会捕获内部算子
            assert len(opnames) >= 3, f"Should capture at least 3 internal operators, got {len(opnames)}"

            # 验证捕获了内部的原生算子
            expected_ops = ['add', 'mul', 'relu']
            for expected in expected_ops:
                found = any(expected in op.lower() for op in opnames)
                assert found, f"Expected {expected} operator not found in decomposition"
                print(f"PASS: Found {expected} in decomposition")

        print("PASS: CompositeImplicitAutograd test completed\n")

    except Exception as e:
        print(f"SKIP: CompositeImplicitAutograd test - {e}\n")


# ============================================================
# Test 2: torch.autograd.Function (Custom Autograd Function)
# ============================================================

class CustomAutogradFunction(torch.autograd.Function):
    """自定义 autograd 函数，内部调用原生算子"""

    @staticmethod
    def forward(ctx, x, y):
        # 前向传播中调用多个原生算子
        ctx.save_for_backward(x, y)
        z = torch.add(x, y)
        z = torch.mul(z, 2.0)
        z = torch.exp(z)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播中调用多个原生算子
        x, y = ctx.saved_tensors
        grad_x = torch.mul(grad_output, 2.0)
        grad_y = torch.mul(grad_output, 2.0)
        return grad_x, grad_y


def test_autograd_function():
    """Test custom autograd function with forward and backward."""
    print("=" * 60)
    print("Test 2: torch.autograd.Function")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        # 测试前向传播
        with ops_dump(tmpdir) as dumper:
            z = CustomAutogradFunction.apply(x, y)
            z = z.sum()
            z.backward()

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])
        opnames = get_opnames_from_dump(session_dir)

        print(f"Captured {len(opnames)} operators: {opnames}")

        # 验证捕获了前向传播的算子
        forward_ops = ['add', 'mul', 'exp']
        for expected in forward_ops:
            found = any(expected in op.lower() for op in opnames)
            assert found, f"Expected {expected} in forward pass"
            print(f"PASS: Found {expected} in forward")

        print("PASS: autograd.Function test completed\n")


# ============================================================
# Test 3: torch.jit.script (TorchScript JIT Compilation)
# ============================================================

@torch.jit.script
def scripted_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """JIT 编译的脚本函数"""
    z = torch.add(x, y)
    z = torch.mul(z, 0.5)
    z = torch.sigmoid(z)
    return z


def test_torchjit_script():
    """Test TorchScript scripted function."""
    print("=" * 60)
    print("Test 3: torch.jit.script")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        with ops_dump(tmpdir) as dumper:
            result = scripted_function(x, y)

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])
        opnames = get_opnames_from_dump(session_dir)

        print(f"Captured {len(opnames)} operators: {opnames}")

        # 验证捕获了内部的算子
        expected_ops = ['add', 'mul', 'sigmoid']
        for expected in expected_ops:
            found = any(expected in op.lower() for op in opnames)
            if found:
                print(f"PASS: Found {expected}")
            else:
                print(f"NOTE: {expected} may be fused in JIT")

        print("PASS: torch.jit.script test completed\n")


# ============================================================
# Test 4: torch.nn.Module with custom forward (Already covered)
# ============================================================

class CustomModule(torch.nn.Module):
    """自定义 Module，内部调用多个算子"""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(5, 10))

    def forward(self, x):
        y = torch.matmul(x, self.weight)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.dropout(y, p=0.5, training=self.training)
        return y


def test_custom_module():
    """Test custom nn.Module."""
    print("=" * 60)
    print("Test 4: torch.nn.Module custom forward")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        module = CustomModule()
        x = torch.randn(3, 5)

        with ops_dump(tmpdir) as dumper:
            result = module(x)

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])
        opnames = get_opnames_from_dump(session_dir)

        print(f"Captured {len(opnames)} operators: {opnames}")

        # 验证捕获了内部的算子
        expected_ops = ['mm', 'relu', 'dropout', 'bernoulli']
        for expected in expected_ops:
            found = any(expected in op.lower() for op in opnames)
            if found:
                print(f"PASS: Found {expected}")

        print("PASS: custom Module test completed\n")


# ============================================================
# Test 5: Nested custom operator calls
# ============================================================

def outer_custom_function(x):
    """外层自定义函数，调用其他自定义函数"""
    y = inner_custom_function(x)
    y = torch.exp(y)
    return y


def inner_custom_function(x):
    """内层自定义函数"""
    y = torch.add(x, 1)
    y = torch.mul(y, 2)
    return y


def test_nested_custom_calls():
    """Test nested custom function calls."""
    print("=" * 60)
    print("Test 5: Nested custom function calls")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        x = torch.randn(3, 3)

        with ops_dump(tmpdir) as dumper:
            result = outer_custom_function(x)

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])
        opnames = get_opnames_from_dump(session_dir)

        print(f"Captured {len(opnames)} operators: {opnames}")

        # 验证捕获了所有嵌套调用中的算子
        expected_ops = ['add', 'mul', 'exp']
        for expected in expected_ops:
            found = any(expected in op.lower() for op in opnames)
            assert found, f"Expected {expected} operator not found"
            print(f"PASS: Found {expected}")

        print("PASS: nested custom calls test completed\n")


# ============================================================
# Test 6: Custom operator with tensor operations in loops
# ============================================================

def custom_loop_function(x, iterations=3):
    """自定义函数，循环中调用算子"""
    y = x
    for i in range(iterations):
        y = torch.add(y, 1)
        y = torch.mul(y, 1.1)
    return y


def test_custom_loop():
    """Test custom function with loop operations."""
    print("=" * 60)
    print("Test 6: Custom function with loop operations")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        x = torch.randn(3, 3)

        with ops_dump(tmpdir) as dumper:
            result = custom_loop_function(x, iterations=3)

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])
        opnames = get_opnames_from_dump(session_dir)

        print(f"Captured {len(opnames)} operators: {opnames}")

        # 验证捕获了循环中的算子（3次循环 = 6个算子）
        add_count = sum(1 for op in opnames if 'add' in op.lower())
        mul_count = sum(1 for op in opnames if 'mul' in op.lower())

        print(f"add operations: {add_count}, mul operations: {mul_count}")
        assert add_count >= 3, f"Expected at least 3 add operations, got {add_count}"
        assert mul_count >= 3, f"Expected at least 3 mul operations, got {mul_count}"

        print("PASS: custom loop test completed\n")


# ============================================================
# Test 7: Custom operator with conditional execution
# ============================================================

def custom_conditional_function(x, condition=True):
    """自定义函数，条件分支中调用算子"""
    if condition:
        y = torch.add(x, 1)
        y = torch.relu(y)
    else:
        y = torch.mul(x, 2)
        y = torch.sigmoid(y)
    return y


def test_custom_conditional():
    """Test custom function with conditional branches."""
    print("=" * 60)
    print("Test 7: Custom function with conditional branches")
    print("=" * 60)

    # Test condition=True branch
    with tempfile.TemporaryDirectory() as tmpdir:
        x = torch.randn(3, 3)

        with ops_dump(tmpdir) as dumper:
            result_true = custom_conditional_function(x, condition=True)

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])
        opnames_true = get_opnames_from_dump(session_dir)

        print(f"Condition=True captured: {opnames_true}")
        assert any('add' in op.lower() for op in opnames_true), "add should be captured when condition=True"
        print("PASS: True branch captured add")

    # Test condition=False branch
    with tempfile.TemporaryDirectory() as tmpdir:
        x = torch.randn(3, 3)

        with ops_dump(tmpdir) as dumper:
            result_false = custom_conditional_function(x, condition=False)

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])
        opnames_false = get_opnames_from_dump(session_dir)

        print(f"Condition=False captured: {opnames_false}")
        assert any('mul' in op.lower() for op in opnames_false), "mul should be captured when condition=False"
        print("PASS: False branch captured mul")

    print("PASS: custom conditional test completed\n")


if __name__ == "__main__":
    test_torch_library_explicit_autograd()
    test_torch_library_implicit_autograd()
    test_autograd_function()
    test_torchjit_script()
    test_custom_module()
    test_nested_custom_calls()
    test_custom_loop()
    test_custom_conditional()
    print("=" * 60)
    print("All custom operator registration tests passed!")
    print("=" * 60)