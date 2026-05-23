# Cache 存储层改用 Tensor 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 cache 存储块从 numpy 改为 tensor，支持 BFloat16 等特殊类型

**Architecture:** 修改 `cache.py` 的四个函数/方法，numpy 对象转为 tensor 存储，加载时根据 `type` 字段还原为原始类型

**Tech Stack:** PyTorch, numpy, hashlib

---

## 文件结构

| 文件 | 操作 | 说明 |
|------|------|------|
| `acc/cache.py` | 修改 | 核心存储逻辑 |
| `tests/test_cache.py` | 修改 | 添加 numpy 和 BFloat16 测试 |

---

### Task 1: 添加 numpy roundtrip 测试

**Files:**
- Modify: `tests/test_cache.py:98`

- [ ] **Step 1: 添加 numpy 加载测试**

在 `test_load_tensor` 函数后添加：

```python
def test_load_numpy():
    print("Test: load numpy from CacheEntry")
    with tempfile.TemporaryDirectory() as tmpdir:
        io_writer = IOWriter(enable_async=False)
        mgr = CacheManager(tmpdir, io_writer)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        entry = mgr.save(a)
        restored = mgr.load(entry)
        assert isinstance(restored, np.ndarray)
        assert np.equal(restored, a).all()
    print("  PASS")
```

并在 `main()` 函数中添加调用：

```python
def main():
    print("ACC - CacheManager Unit Tests\n")
    test_cache_entry()
    test_cache_entry_from_obj()
    test_cache_entry_to_obj()
    test_save_tensor()
    test_save_numpy()
    test_save_scalar()
    test_load_tensor()
    test_load_numpy()
    test_save_load_nested()
    test_different_tensors_different_hash()
    test_identical_tensors_same_hash()
    test_same_storage_different_shape()
    print("\nAll cache tests passed.")
```

- [ ] **Step 2: 运行测试验证添加成功**

Run: `python tests/test_cache.py`
Expected: 所有测试 PASS

---

### Task 2: 添加 BFloat16 测试

**Files:**
- Modify: `tests/test_cache.py`

- [ ] **Step 1: 添加 BFloat16 测试**

在 `test_load_numpy` 后添加：

```python
def test_bfloat16_tensor():
    print("Test: BFloat16 tensor save/load")
    with tempfile.TemporaryDirectory() as tmpdir:
        io_writer = IOWriter(enable_async=False)
        mgr = CacheManager(tmpdir, io_writer)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        entry = mgr.save(t)
        restored = mgr.load(entry)
        assert isinstance(restored, torch.Tensor)
        assert restored.dtype == torch.bfloat16
        assert torch.equal(restored, t)
    print("  PASS")
```

并在 `main()` 中添加 `test_bfloat16_tensor()` 调用。

- [ ] **Step 2: 运行测试验证 BFloat16 失败（预期）**

Run: `python tests/test_cache.py`
Expected: BFloat16 测试 FAIL（TypeError: Got unsupported ScalarType BFloat16）

---

### Task 3: 修改 `_extract_storage` 返回 tensor

**Files:**
- Modify: `acc/cache.py:46-54`

- [ ] **Step 1: 修改 `_extract_storage` 函数**

```python
def _extract_storage(obj: Any) -> torch.Tensor:
    """从 tensor/numpy 中提取存储块（tensor）"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().contiguous().cpu()
    # numpy 转 tensor
    return torch.from_numpy(obj).contiguous().cpu()
```

- [ ] **Step 2: 运行测试验证**

Run: `python tests/test_cache.py`
Expected: BFloat16 测试仍 FAIL（下一步修复）

---

### Task 4: 修改 `_compute_hash` 处理 tensor

**Files:**
- Modify: `acc/cache.py:57-59`

- [ ] **Step 1: 修改 `_compute_hash` 函数**

```python
def _compute_hash(storage: torch.Tensor) -> str:
    """计算 tensor 存储块的 BLAKE2b 哈希"""
    return hashlib.blake2b(storage.numpy().tobytes(), digest_size=32).hexdigest()
```

- [ ] **Step 2: 运行测试验证**

Run: `python tests/test_cache.py`
Expected: BFloat16 测试 FAIL（hash 步骤可能报错）

---

### Task 5: 修改 `_load_cache_map` 类型

**Files:**
- Modify: `acc/cache.py:69`

- [ ] **Step 1: 修改 `_load_cache_map` 类型注解**

```python
self._load_cache_map: Dict[str, torch.Tensor] = {}  # cache_id -> 存储块
```

---

### Task 6: 修改 `CacheEntry.to_obj` 接收 tensor

**Files:**
- Modify: `acc/cache.py:33-43`

- [ ] **Step 1: 修改 `to_obj` 方法**

```python
def to_obj(self, storage: torch.Tensor) -> Any:
    """从 tensor 存储块重建 tensor/numpy"""
    if self.type == 'tensor':
        t = storage
        if list(t.shape) != self.shape:
            t = t.reshape(self.shape)
        return t
    else:
        # numpy 类型
        arr = storage.numpy()
        if list(arr.shape) != self.shape:
            arr = arr.reshape(self.shape)
        return arr
```

- [ ] **Step 2: 运行所有 cache 测试**

Run: `python tests/test_cache.py`
Expected: 所有测试 PASS

---

### Task 7: 运行完整测试套件

**Files:**
- None

- [ ] **Step 1: 运行 pytest**

Run: `python -m pytest tests/ -v`
Expected: 所有测试 PASS

---

### Task 8: 运行 run.sh 验证

**Files:**
- None

- [ ] **Step 1: 执行 nanoGPT 训练脚本**

Run: `cd /mnt/c/Users/uh/code/work/train-0116/nanoGPT && timeout 120 python train.py config/train_shakespeare_char.py 2>&1`
Expected: 脚本运行成功，无 TypeError

---

### Task 9: 提交代码

**Files:**
- None

- [ ] **Step 1: Git commit**

```bash
git add acc/cache.py tests/test_cache.py
git commit -m "refactor: cache storage uses tensor instead of numpy

- _extract_storage returns torch.Tensor
- _compute_hash handles tensor input
- CacheEntry.to_obj takes tensor and restores original type
- Supports BFloat16 and other special dtypes

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```