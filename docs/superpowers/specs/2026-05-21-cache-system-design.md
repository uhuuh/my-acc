# Cache System Design

## Overview

为PyTorch算子dump工具设计cache机制，用于优化tensor/numpy的存储和加载，避免重复存储相同内容。

## Requirements Summary

1. Cache机制：基于内容哈希缓存tensor/numpy
2. 存储结构：独立文件存储在storage目录
3. 序列化改进：分离元数据(JSON)和数据(PKL)，按需加载
4. 输入输出重组：inputs分为args/kwargs，outputs保持list
5. 比较改进：nan/inf描述、进度显示、kwargs带key信息

## Architecture

### Module Structure

```
acc/
├── __init__.py          # 导出 ops_dump, ops_comp, OperatorRecord, SerializationSession
├── cache.py             # CacheEntry, CacheManager (内部模块)
├── serialization.py     # OperatorDump, SerializationSession, 序列化/反序列化
├── dump.py              # ops_dump context manager (简化)
├── comp.py              # ops_comp, 比较流程
├── comparators.py       # 各Comparator类 (增加nan/inf描述)
├── formatting.py        # 格式化函数 (增加format_eta)
```

## Component Details

### 1. Cache System (cache.py)

#### CacheEntry

```python
@dataclass
class CacheEntry:
    """缓存条目的元信息"""
    cache_id: str          # 内容哈希值（唯一标识，同时作为存储文件名）
    type: str              # 'tensor' 或 'numpy'
    dtype: str             # 数据类型如 'float32', 'int64'
    shape: List[int]       # 形状
```

#### CacheManager

```python
class CacheManager:
    def __init__(self, storage_dir: str, enable_cache: bool = True):
        self.storage_dir = storage_dir
        self.enable_cache = enable_cache
        self._cached_ids: Set[str] = set()  # 已缓存的content_hash

    def get_or_cache(self, obj: Any) -> Any:
        """
        输入任意对象，返回：
        - tensor/numpy + enable_cache=True：返回CacheEntry（新缓存或已缓存）
        - tensor/numpy + enable_cache=False：直接返回原对象（不缓存）
        - 其他对象：直接返回原对象
        """

    def resolve(self, obj: Any) -> Any:
        """
        输入任意对象，返回：
        - CacheEntry：从storage重建tensor/numpy并返回
        - 其他对象：直接返回
        """

    def _compute_content_hash(self, obj) -> str:
        """使用BLAKE2计算tensor/numpy内容的哈希值"""

    def _save_to_storage(self, obj, cache_id: str):
        """保存tensor/numpy到 storage/{cache_id}.pkl"""
```

**哈希计算策略**
- 使用 `hashlib.blake2b` + `tensor.numpy().tobytes()` 直接获取连续内存块
- BLAKE2比MD5更快且更安全，是现代推荐的哈希算法
- cache_id即为content_hash，无需映射关系

### 2. Serialization Layer (serialization.py)

#### OperatorRecord

```python
@dataclass
class OperatorRecord:
    sequence: int
    filepath: str
    filename: str
    function: str
    lineno: int
    opname: str
    call_stack: List[Dict]
    args: List[Any] = field(default_factory=list)      # positional args
    kwargs: Dict[str, Any] = field(default_factory=dict)  # keyword args
    outputs: List[Any] = field(default_factory=list)   # outputs (list形式)
```

#### SerializationSession

```python
class SerializationSession:
    """管理单个序列化会话的状态和保存，内部集成CacheManager"""

    def __init__(self, dump_path: str, max_tensor_size_mb: int = 10240, enable_cache: bool = True):
        self.session_dir: str = None
        self.sequence: int = 0
        self.max_tensor_size_mb: int = max_tensor_size_mb
        self._cache_manager: CacheManager = None  # 内部使用，对外隐藏
        self._enable_cache: bool = enable_cache

    def start(self) -> str:
        """创建会话目录和storage子目录，初始化CacheManager"""

    def save_operation(self, func, filepath, filename, function, lineno,
                       args, kwargs, outputs) -> int:
        """保存单个算子dump，返回sequence"""

    def end(self):
        """结束会话，打印总结"""

    @staticmethod
    def load_metadata(json_path: str) -> OperatorRecord:
        """加载JSON元数据，不含inputs/outputs"""

    @staticmethod
    def load_data(pkl_path: str, storage_dir: str) -> Tuple[Dict, List]:
        """加载PKL数据，自动从storage目录解析CacheEntry为实际对象"""
```

**存储结构**
```
session_dir/
├── storage/
│   ├── {content_hash_1}.pkl
│   ├── {content_hash_2}.pkl
│   └── ...
├── 000001__{filename}__{function}__{opname}.json  (元数据)
├── 000001__{filename}__{function}__{opname}.pkl   (数据)
├── 000002__...
└── ...
```

**PKL数据结构**
```python
{
    'inputs': {
        'args': [arg0, arg1, ...],      # tensor/numpy已替换为CacheEntry
        'kwargs': {'key1': val1, ...}   # tensor/numpy已替换为CacheEntry
    },
    'outputs': [out0, out1, ...]        # tensor/numpy已替换为CacheEntry
}
```

### 3. Dump Layer (dump.py)

简化为只负责捕获：

```python
class ops_dump(TorchDispatchMode):
    def __init__(self, dump_path: str, max_tensor_size_mb: int = 10240, enable_cache: bool = True):
        self.session = SerializationSession(dump_path, max_tensor_size_mb, enable_cache)

    def __enter__(self):
        self.session.start()
        return super().__enter__()

    def __torch_dispatch__(self, func, types, args, kwargs):
        result = func(*args, **kwargs)
        # 捕获调用栈信息
        self.session.save_operation(func, filepath, filename, ..., args, kwargs, result)
        return result

    def __exit__(self, ...):
        self.session.end()
        return super().__exit__()
```

### 4. Comparison Layer (comp.py)

#### 比较流程

```python
def ops_comp(dump_dir_a: str, dump_dir_b: str):
    # 1. 加载所有元数据（不加载实际数据）
    dumps_a = load_all_metadata(dump_dir_a)
    dumps_b = load_all_metadata(dump_dir_b)

    # 2. 准备storage目录路径（用于load_data时解析cache）
    storage_dir_a = os.path.join(dump_dir_a, 'storage')
    storage_dir_b = os.path.join(dump_dir_b, 'storage')

    # 3. LCS匹配（只用元数据）
    matched_pairs = find_lcs_matches(dumps_a, dumps_b)

    # 4. 比较matched pairs（按需加载实际数据）
    compare_matched_pairs(dumps_a, dumps_b, matched_pairs,
                          storage_dir_a, storage_dir_b)
```

#### 进度显示

```python
def compare_matched_pairs(dumps_a, dumps_b, matched_pairs, storage_dir_a, storage_dir_b):
    total = len(matched_pairs)
    start_time = time.time()

    for idx, (idx_a, idx_b) in enumerate(matched_pairs, 1):
        # ETA估算
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        eta_seconds = avg_time * (total - idx)

        print(f"[COMPARE {idx}/{total} | ETA: {format_eta(eta_seconds)}]")

        # 按需加载实际数据并比较（SerializationSession.load_data内部处理cache）
        inputs_a, outputs_a = SerializationSession.load_data(pkl_path_a, storage_dir_a)
        inputs_b, outputs_b = SerializationSession.load_data(pkl_path_b, storage_dir_b)
        ...

def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m{int(seconds%60)}s"
    else:
        return f"{int(seconds/3600)}h{int(seconds%3600/60)}m"
```

#### 输出格式

```python
# args比较
Inputs.args[0] | tensor(dtype=float32, shape=[2,3], nan=False, inf=False) | tensor(...) | exact_match=True

# kwargs比较
Inputs.kwargs[alpha] | float(0.5) | float(0.5) | exact_match=True

# outputs比较
Outputs[0] | tensor(...) | tensor(...) | ...
```

### 5. Comparators (comparators.py)

#### TensorComparator 类型描述增强

```python
class TensorComparator(ElementComparator):
    def get_type_info(self) -> Tuple[str, str]:
        # 检测 nan/inf/-inf
        a_nan = torch.isnan(self.a).any().item()
        a_inf = torch.isinf(self.a).any().item()
        a_neg_inf = a_inf and (self.a < 0).any().item()

        b_nan = torch.isnan(self.b).any().item()
        b_inf = torch.isinf(self.b).any().item()
        b_neg_inf = b_inf and (self.b < 0).any().item()

        # 构建描述字符串
        desc_a = f"tensor(dtype={dtype_a}, shape={shape_a}, nan={a_nan}, inf={a_inf}, neg_inf={a_neg_inf})"
        desc_b = f"tensor(dtype={dtype_b}, shape={shape_b}, nan={b_nan}, inf={b_inf}, neg_inf={b_neg_inf})"

        return desc_a, desc_b
```

NumpyComparator 同样处理。

## Migration Strategy

1. **新增cache.py** - 实现CacheEntry和CacheManager（内部模块，enable_cache参数）
2. **改造serialization.py** - 新增SerializationSession，调整OperatorDump结构，实现save/load接口
3. **简化dump.py** - 移除序列化逻辑，改用SerializationSession
4. **改造comp.py** - 实现按需加载、进度显示、args/kwargs分离比较（使用SerializationSession.load_data）
5. **改造comparators.py** - 增加nan/inf描述
6. **更新__init__.py** - 导出SerializationSession（CacheManager作为内部模块不导出）

## Test Coverage

- CacheManager: get_or_cache, resolve, content_hash计算, enable_cache参数
- SerializationSession: start, save_operation, end, load_metadata, load_data
- 比较流程: args/kwargs分离、进度显示
- Comparators: nan/inf描述正确性