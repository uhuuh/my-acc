# Cache 存储层改用 Tensor 设计

## 背景

当前 `cache.py` 使用 numpy 作为存储块，但 BFloat16 等类型不被 numpy 支持，导致运行时报错。

## 目标

将存储块统一改为 tensor，numpy 对象先转为 tensor 存储，加载时根据 `type` 字段还原为原始类型。

## 设计

### 1. `_extract_storage` 函数

返回 tensor 而非 numpy：

```python
def _extract_storage(obj: Any) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj.detach().contiguous().cpu()
    return torch.from_numpy(obj).contiguous().cpu()
```

### 2. `_compute_hash` 函数

处理 tensor：

```python
def _compute_hash(storage: torch.Tensor) -> str:
    return hashlib.blake2b(storage.numpy().tobytes(), digest_size=32).hexdigest()
```

### 3. `_load_cache_map` 类型

```python
self._load_cache_map: Dict[str, torch.Tensor] = {}
```

### 4. `CacheEntry.to_obj` 方法

根据 type 字段还原：

```python
def to_obj(self, storage: torch.Tensor) -> Any:
    if self.type == 'tensor':
        t = storage
        if list(t.shape) != self.shape:
            t = t.reshape(self.shape)
        return t
    else:
        arr = storage.numpy()
        if list(arr.shape) != self.shape:
            arr = arr.reshape(self.shape)
        return arr
```

## 不改动部分

- `CacheEntry.type` 字段保留，区分 tensor/numpy
- `comparators.py` 的 `get_type_info` 保留 numpy 显示
- `comparators.py` 的 `compare` 已处理 numpy→tensor
- `serialization.py` 无需改动

## 验证

执行 `/mnt/c/Users/uh/code/work/train-0116/nanoGPT/my/run.sh` 正确运行。