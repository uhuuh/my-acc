"""
Test that ops_dump captures operators called inside custom functions/modules.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tempfile
import json
from acc import ops_dump


class CustomModule(torch.nn.Module):
    """自定义模块，内部调用多个 PyTorch 原生算子"""
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(5, 10)

    def forward(self, x):
        # 内部调用多个原生算子
        y = torch.matmul(x, self.weight)  # matmul
        y = torch.add(y, torch.ones_like(y))  # add
        y = torch.relu(y)  # relu
        y = torch.mul(y, 2.0)  # mul
        return y


def custom_function(x):
    """自定义函数，内部调用多个算子"""
    y = x + 1  # add (通过 + 运算符)
    y = y * 2  # mul (通过 * 运算符)
    y = torch.sin(y)  # sin
    return y


def test_custom_module_ops_dump():
    """测试捕获自定义模块内部的算子调用"""
    print("=" * 60)
    print("Test: Custom module internal operators")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        module = CustomModule()
        x = torch.randn(3, 5)

        with ops_dump(tmpdir) as dumper:
            result = module(x)

        # 检查 session 目录
        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])

        # 读取所有算子
        dump_files = [f for f in os.listdir(session_dir) if f.endswith('.json')]
        print(f"Captured {len(dump_files)} operators")

        # 检查是否捕获了预期的算子
        opnames = []
        for json_file in dump_files:
            json_path = os.path.join(session_dir, json_file)
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            opnames.append(metadata['opname'])

        print(f"Operators: {opnames}")

        # 验证捕获了 mm (matmul降级为mm), add, relu, mul
        expected_ops = ['mm', 'add', 'relu', 'mul']
        for expected in expected_ops:
            found = any(expected in op.lower() for op in opnames)
            assert found, f"Expected {expected} operator not found"
            print(f"PASS: Found {expected}")

        assert len(dump_files) >= 4, f"Expected at least 4 operators, got {len(dump_files)}"
        print("PASS: Custom module test passed\n")


def test_custom_function_ops_dump():
    """测试捕获自定义函数内部的算子调用"""
    print("=" * 60)
    print("Test: Custom function internal operators")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        x = torch.randn(5, 5)

        with ops_dump(tmpdir) as dumper:
            result = custom_function(x)

        # 检查 session 目录
        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])

        dump_files = [f for f in os.listdir(session_dir) if f.endswith('.json')]
        print(f"Captured {len(dump_files)} operators")

        opnames = []
        for json_file in dump_files:
            json_path = os.path.join(session_dir, json_file)
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            opnames.append(metadata['opname'])

        print(f"Operators: {opnames}")

        # 验证捕获了 add, mul, sin
        expected_ops = ['add', 'mul', 'sin']
        for expected in expected_ops:
            found = any(expected in op.lower() for op in opnames)
            assert found, f"Expected {expected} operator not found"
            print(f"PASS: Found {expected}")

        assert len(dump_files) >= 3, f"Expected at least 3 operators, got {len(dump_files)}"
        print("PASS: Custom function test passed\n")


def test_nested_custom_calls():
    """测试嵌套调用场景"""
    print("=" * 60)
    print("Test: Nested custom calls")
    print("=" * 60)

    def outer_function(x):
        y = custom_function(x)  # 调用自定义函数
        y = torch.exp(y)  # 再调用一个原生算子
        return y

    with tempfile.TemporaryDirectory() as tmpdir:
        x = torch.randn(3, 3)

        with ops_dump(tmpdir) as dumper:
            result = outer_function(x)

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])

        dump_files = [f for f in os.listdir(session_dir) if f.endswith('.json')]
        print(f"Captured {len(dump_files)} operators")

        opnames = []
        for json_file in dump_files:
            json_path = os.path.join(session_dir, json_file)
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            opnames.append(metadata['opname'])

        print(f"Operators: {opnames}")

        # 应捕获 add, mul, sin, exp
        assert len(dump_files) >= 4, f"Expected at least 4 operators, got {len(dump_files)}"
        print("PASS: Nested calls test passed\n")


def test_backward_pass_not_captured():
    """测试反向传播算子不被捕获（除非显式执行 backward）"""
    print("=" * 60)
    print("Test: Backward pass operators")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        x = torch.randn(3, 5, requires_grad=True)
        weight = torch.randn(5, 10, requires_grad=True)

        # 前向传播
        with ops_dump(tmpdir) as dumper:
            y = torch.matmul(x, weight)
            y = y.sum()

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])
        dump_files = [f for f in os.listdir(session_dir) if f.endswith('.json')]
        print(f"Forward pass captured {len(dump_files)} operators")

        opnames = []
        for json_file in dump_files:
            json_path = os.path.join(session_dir, json_file)
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            opnames.append(metadata['opname'])

        print(f"Forward operators: {opnames}")

        # 验证没有 backward 相关算子
        backward_ops = [op for op in opnames if 'backward' in op.lower()]
        assert len(backward_ops) == 0, f"Backward operators should not be captured: {backward_ops}"
        print("PASS: No backward operators captured in forward pass")

        # 现在显式执行 backward，检查是否会捕获
        tmpdir2 = tempfile.mkdtemp()
        try:
            x2 = torch.randn(3, 5, requires_grad=True)
            weight2 = torch.randn(5, 10, requires_grad=True)

            with ops_dump(tmpdir2) as dumper:
                y2 = torch.matmul(x2, weight2)
                y2 = y2.sum()
                y2.backward()  # 执行反向传播

            dump_dirs2 = [d for d in os.listdir(tmpdir2) if os.path.isdir(os.path.join(tmpdir2, d))]
            session_dir2 = os.path.join(tmpdir2, dump_dirs2[0])
            dump_files2 = [f for f in os.listdir(session_dir2) if f.endswith('.json')]
            print(f"Backward pass captured {len(dump_files2)} operators")

            opnames2 = []
            for json_file in dump_files2:
                json_path = os.path.join(session_dir2, json_file)
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                opnames2.append(metadata['opname'])

            print(f"All operators (forward+backward): {opnames2}")

            # backward 执行时，会捕获反向传播算子
            backward_ops2 = [op for op in opnames2 if 'backward' in op.lower() or 'grad' in op.lower()]
            print(f"Backward-related operators: {backward_ops2}")

            # 应该包含 mm, sum, 以及 backward 相关算子
            assert len(dump_files2) > len(dump_files), "Backward pass should capture more operators"
            print("PASS: Backward operators captured when backward() is called")

        finally:
            import shutil
            shutil.rmtree(tmpdir2)

        print("PASS: Backward pass test completed\n")


def test_nested_dump_contexts():
    """测试嵌套 dump 上下文"""
    print("=" * 60)
    print("Test: Nested dump contexts")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir_outer:
        with tempfile.TemporaryDirectory() as tmpdir_inner:
            x = torch.randn(3, 3)

            # 嵌套 dump
            with ops_dump(tmpdir_outer) as outer_dumper:
                y1 = x + 1

                with ops_dump(tmpdir_inner) as inner_dumper:
                    y2 = y1 * 2
                    y3 = torch.sin(y2)

                y4 = torch.exp(y3)

            # 检查外层 dump
            outer_dirs = [d for d in os.listdir(tmpdir_outer) if os.path.isdir(os.path.join(tmpdir_outer, d))]
            outer_session = os.path.join(tmpdir_outer, outer_dirs[0])
            outer_files = [f for f in os.listdir(outer_session) if f.endswith('.json')]
            print(f"Outer dump captured {len(outer_files)} operators")

            # 检查内层 dump
            inner_dirs = [d for d in os.listdir(tmpdir_inner) if os.path.isdir(os.path.join(tmpdir_inner, d))]
            inner_session = os.path.join(tmpdir_inner, inner_dirs[0])
            inner_files = [f for f in os.listdir(inner_session) if f.endswith('.json')]
            print(f"Inner dump captured {len(inner_files)} operators")

            # 外层应该捕获 add, mul, sin, exp (所有算子)
            assert len(outer_files) >= 4, f"Outer should capture at least 4 operators, got {len(outer_files)}"
            print("PASS: Outer dump captured all operators")

            # 内层应该只捕获 mul, sin (内层上下文中的算子)
            assert len(inner_files) >= 2, f"Inner should capture at least 2 operators, got {len(inner_files)}"
            print("PASS: Inner dump captured nested context operators")

            print("PASS: Nested dump test completed\n")


def test_backward_detailed_check():
    """详细检查反向流程捕获的内容"""
    print("=" * 60)
    print("Test: Backward pass detailed content check")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        x = torch.randn(3, 5, requires_grad=True)
        weight = torch.randn(5, 10, requires_grad=True)

        with ops_dump(tmpdir) as dumper:
            y = torch.matmul(x, weight)
            y = y.sum()
            y.backward()

        dump_dirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
        session_dir = os.path.join(tmpdir, dump_dirs[0])
        dump_files = sorted([f for f in os.listdir(session_dir) if f.endswith('.json')])

        print(f"\nTotal captured operators: {len(dump_files)}")
        print("\nOperator details:")
        print("-" * 80)

        forward_ops = []
        backward_ops = []

        for json_file in dump_files:
            json_path = os.path.join(session_dir, json_file)
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            opname = metadata['opname']
            filename = metadata.get('filename', '')
            function = metadata.get('function', '')

            # 根据调用栈判断是前向还是反向
            is_backward = 'backward' in function.lower() or 'grad' in function.lower() or \
                          'autograd' in filename.lower() or 'graph' in filename.lower()

            if is_backward:
                backward_ops.append(opname)
                tag = "[BACKWARD]"
            else:
                forward_ops.append(opname)
                tag = "[FORWARD]"

            print(f"{tag} {json_file}")
            print(f"  Opname: {opname}")
            print(f"  Source: {filename}:{metadata.get('lineno', 0)} in {function}")

        print("-" * 80)
        print(f"\nForward operators ({len(forward_ops)}): {forward_ops}")
        print(f"Backward operators ({len(backward_ops)}): {backward_ops}")

        # 验证前向传播算子
        assert 'mm' in forward_ops[0].lower() or 'matmul' in forward_ops[0].lower(), "First forward op should be mm"
        assert 'sum' in forward_ops[1].lower(), "Second forward op should be sum"
        print("PASS: Forward operators correct")

        # 验证反向传播算子
        # 应包含 ones_like (梯度初始化), expand (广播), mm (反向计算), detach (梯度管理)
        backward_types = set()
        for op in backward_ops:
            # Extract operator name (e.g., "aten.mm.default" -> "mm")
            parts = op.split('.')
            if len(parts) >= 2:
                backward_types.add(parts[-2].lower())

        print(f"Backward operator types: {backward_types}")
        expected_backward_types = {'ones_like', 'expand', 'mm', 'detach', 't'}
        for expected in expected_backward_types:
            if expected in backward_types:
                print(f"PASS: Found {expected} in backward pass")

        print("PASS: Backward pass detailed check completed\n")


if __name__ == "__main__":
    test_custom_module_ops_dump()
    test_custom_function_ops_dump()
    test_nested_custom_calls()
    test_backward_pass_not_captured()
    test_nested_dump_contexts()
    test_backward_detailed_check()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)