# PyTorch Operator Dump & Precision Comparison Tool Design

**Date:** 2026-05-18
**Status:** Design Approved

## Overview

A single-file implementation providing two tools for PyTorch operator debugging:
- **OperatorDumper** (class): Capture operator calls with full context
- **compare_operator_dumps()** (function): Compare two dump sessions with precision analysis

## Architecture

```
operator_tools.py
├── OperatorDumper (context manager + decorator)
│   ├── Uses TorchDispatchMode to intercept all torch ops
│   ├── Creates session directory (path/pid/timestamp)
│   └── Dumps each op call to separate pickle file
│
└── compare_operator_dumps() (standalone function)
    ├── Loads two dump directories
    ├── Finds LCS of operator sequences
    └── Reports comparison metrics
```

## OperatorDumper Design

### API

```python
# Context manager
with OperatorDumper("/path/to/dumps") as dumper:
    model(input)

# Decorator
@OperatorDumper("/path/to/dumps")
def run_model(input):
    return model(input)
```

### Internal Flow

1. On `__enter__` or `__call__`: create directory `{path}/{pid}_{YYYYMMDD_HHMMSS}/`
2. On each torch op: capture via `TorchDispatchMode.__torch_dispatch__`
3. Extract: filename, line number, operator name, inputs, outputs, full call stack
4. Convert tensors to CPU, serialize via pickle
5. Write to `{sequence:04d}_{filename}_{opname}.pkl`
6. Log each dump operation

### Dump File Structure

Each dump file contains:
```python
{
    'sequence': int,              # Sequential order (0001, 0002, ...)
    'filename': str,              # Source file name
    'lineno': int,                # Line number in source
    'opname': str,                # Operator name
    'call_stack': str,            # Full traceback string
    'inputs': [...],              # Serialized inputs with tensors on CPU
    'outputs': [...]              # Serialized outputs with tensors on CPU
}
```

### Session Directory Structure

```
/path/to/dumps/
└── 12345_20260518_200530/      # pid_timestamp
    ├── 0001_my_script_add.pkl
    ├── 0002_my_script_matmul.pkl
    ├── 0003_my_script_relu.pkl
    ...
```

### Dump Logging Format

```
[DUMP] Created session directory: /path/to/dumps/12345_20260518_200530
[DUMP] 0001 | my_script.py:42 | torch.add | saved to 0001_my_script_add.pkl
[DUMP] 0002 | my_script.py:45 | torch.matmul | saved to 0002_my_script_matmul.pkl
...
[DUMP] Session completed: 42 operators dumped to /path/to/dumps/12345_20260518_200530
```

**Format:** `[DUMP] <sequence> | <filename>:<lineno> | <operator_name> | saved to <filename>`

## compare_operator_dumps Design

### API

```python
compare_operator_dumps("/path/to/dump1", "/path/to/dump2")
```

### Comparison Flow

**Phase 1: LCS Matching**
1. Load all `.pkl` files from both directories
2. Sort by sequence number
3. Build signature for each op: `"{filename}::{opname}"` (e.g., "my_script.py::torch.add")
4. Find LCS between two signature sequences
5. Log matching and skipping operators

**Phase 2: Detailed Comparison**
1. For matched pairs: compare inputs and outputs
2. Apply universal comparison flow for each element
3. Log comparison results with metrics

### Universal Comparison Flow

```
1. 检查对应关系
   - 左右是否有对应的元素
   - 数量是否一致

2. 检查类型
   - int, float, tensor, numpy, None, unsupported

3. 根据类型执行比较
   - int/float: 直接比较值
   - None: 比较是否都为None
   - tensor/numpy: dtype → shape → content metrics
   - unsupported: 标记不支持

4. 输出比较结果
   - 包含完整信息
   - 保持格式一致性
```

### Supported Types

| Type | Comparison Strategy |
|------|---------------------|
| **None** | Direct equality check |
| **int** | Direct equality, show values and diff |
| **float** | Direct equality, show values and diff |
| **tensor** | dtype, shape, content metrics |
| **numpy** | Convert to tensor, full comparison |
| **unsupported** | Mark as unsupported (list, tuple, dict, etc.) |

### Tensor Content Metrics

For tensors/numpy arrays:
- `exact_match`: bool - 逐元素是否完全相等
- `match_ratio`: float - 逐元素相等比率 (相等元素数/总元素数)
- `max_err`: float - 最大绝对误差
- `min_err`: float - 最小绝对误差
- `mean_err`: float - 平均绝对误差
- `mse`: float - 均方误差
- `cosine`: float - 余弦相似度

### Comparison Logging Format

**Three-part structure:** `位置信息 | 左type信息 | 右type信息 | 比较细节`

**Examples:**

```
# None - matched
Inputs[0] | None | None | exact_match=True

# int - mismatched
Inputs[3] | int(42) | int(43) | exact_match=False, diff=1

# float - mismatched
Inputs[5] | float(3.14) | float(3.15) | exact_match=False, diff=0.01

# tensor - exact match
Inputs[6] | tensor(dtype=float32, shape=[32,64]) | tensor(dtype=float32, shape=[32,64]) | dtype=match, shape=match, exact_match=True, match_ratio=1.0, max_err=0.0, min_err=0.0, mean_err=0.0, mse=0.0, cosine=1.0

# tensor - precision diff
Inputs[7] | tensor(dtype=float32, shape=[32,64]) | tensor(dtype=float32, shape=[32,64]) | dtype=match, shape=match, exact_match=False, match_ratio=0.95, max_err=2.1e-5, min_err=0.0, mean_err=4.2e-7, mse=1.8e-12, cosine=0.9998

# tensor - dtype mismatch
Inputs[8] | tensor(dtype=float32, shape=[32,64]) | tensor(dtype=float16, shape=[32,64]) | dtype=mismatch, shape=match, content_skipped

# tensor - shape mismatch
Inputs[9] | tensor(dtype=float32, shape=[32,64]) | tensor(dtype=float32, shape=[32,128]) | dtype=match, shape=mismatch, content_skipped

# numpy - precision diff
Inputs[11] | numpy(dtype=float64, shape=[64,128]) | numpy(dtype=float64, shape=[64,128]) | dtype=match, shape=match, exact_match=False, match_ratio=0.87, max_err=1.2e-6, min_err=0.0, mean_err=3.4e-7, mse=3.4e-12, cosine=0.9999

# unsupported type
Inputs[12] | list(len=5) | list(len=6) | type_unsupported

# missing in A
Inputs[13] | <missing> | tensor(dtype=float32, shape=[256,512]) | missing_in_A

# missing in B
Outputs[0] | tensor(dtype=float32, shape=[32,256]) | <missing> | missing_in_B
```

**Format details:**
- Separator: `|` between three major parts
- Detail separator: `,` within comparison details
- Position: `Inputs[i]` or `Outputs[i]` (no outer brackets)
- Type info: tensor/numpy show dtype and shape; int/float show value; None just "None"
- Missing: `<missing>` placeholder
- Grep note: `|` needs escape `\|` in grep

### Phase 1: LCS Logging

```
[LCS] Loading dump A: 42 operators from /path/to/dump1
[LCS] Loading dump B: 38 operators from /path/to/dump2
[LCS] Building operator signatures...
[LCS] Finding longest common subsequence...
[LCS] Matched: 35 operators | A-only: 7 | B-only: 3
[MATCH] A:0001_forward_add <-> B:0001_forward_add
[MATCH] A:0002_matmul <-> B:0002_matmul
[SKIP] A:0003_dropout (no match in B)
[MATCH] A:0004_relu <-> B:0003_relu
[SKIP] B:0005_custom_op (no match in A)
...
```

### Phase 2: Detailed Comparison Logging

```
[COMPARE] Starting detailed comparison of 35 matched pairs...
[COMPARE] 0001_forward_add:
  Inputs[0] | tensor(dtype=float32, shape=[32,64]) | tensor(dtype=float32, shape=[32,64]) | dtype=match, shape=match, exact_match=True, match_ratio=1.0, max_err=0.0, min_err=0.0, mean_err=0.0, mse=0.0, cosine=1.0
  Inputs[1] | tensor(dtype=float32, shape=[64]) | tensor(dtype=float32, shape=[64]) | dtype=match, shape=match, exact_match=True, match_ratio=1.0, max_err=0.0, min_err=0.0, mean_err=0.0, mse=0.0, cosine=1.0
  Outputs[0] | tensor(dtype=float32, shape=[32,64]) | tensor(dtype=float32, shape=[32,64]) | dtype=match, shape=match, exact_match=True, match_ratio=1.0, max_err=0.0, min_err=0.0, mean_err=0.0, mse=0.0, cosine=1.0
[COMPARE] 0002_matmul:
  Inputs[0] | tensor(dtype=float32, shape=[32,64]) | tensor(dtype=float32, shape=[32,64]) | dtype=match, shape=match, exact_match=True, match_ratio=1.0, max_err=0.0, min_err=0.0, mean_err=0.0, mse=0.0, cosine=1.0
  Inputs[1] | tensor(dtype=float32, shape=[64,128]) | tensor(dtype=float32, shape=[64,128]) | dtype=match, shape=match, exact_match=False, match_ratio=0.95, max_err=2.1e-5, min_err=0.0, mean_err=4.2e-7, mse=1.8e-12, cosine=0.9998
  Outputs[0] | tensor(dtype=float32, shape=[32,128]) | tensor(dtype=float32, shape=[32,128]) | dtype=match, shape=match, exact_match=False, match_ratio=0.87, max_err=1.2e-6, min_err=0.0, mean_err=3.4e-7, mse=3.4e-12, cosine=0.9999
[COMPARE] 0003_relu:
  Inputs[0] | tensor(dtype=float32, shape=[32,64]) | tensor(dtype=float32, shape=[32,128]) | dtype=match, shape=mismatch, content_skipped
  Outputs[0] | tensor(dtype=float32, shape=[32,64]) | tensor(dtype=float32, shape=[32,128]) | dtype=match, shape=mismatch, content_skipped
...
[SUMMARY] Total: 35 | Exact match: 28 | Precision diff: 5 | Dtype mismatch: 1 | Shape mismatch: 2 | Input count mismatch: 1 | Output count mismatch: 0
```

## Implementation Details

### Key Dependencies

- `torch.utils._python_dispatch.TorchDispatchMode` - operator interception
- `pickle` - serialization format
- `traceback` - call stack extraction
- `inspect` - source file/line extraction

### File Naming Convention

Dump files: `{sequence:04d}_{filename}_{opname}.pkl`
- sequence: 4-digit zero-padded number
- filename: sanitized source filename (replace `/` with `_`)
- opname: operator name (e.g., `torch.add`, `torch.nn.functional.relu`)

### Tensor Handling

- All tensors converted to CPU before serialization
- Full tensor data saved (not just metadata)
- No truncation in repr

### Edge Cases

1. **Input/Output count mismatch**: Compare by index, log missing elements
2. **Type mismatch**: Log as `type_mismatch` or `dtype_mismatch`
3. **Shape mismatch**: Skip content comparison, log reason
4. **Unsupported types**: Log as `type_unsupported`, show repr
5. **None values**: Handle as supported type
6. **No LCS matches**: Report empty match set

## Summary Statistics

Final summary includes counts of:
- Total matched operators
- Exact matches
- Precision differences
- Dtype mismatches
- Shape mismatches
- Input/output count mismatches