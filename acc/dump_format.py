"""Minimal on-disk dump format and loading helpers."""

import json
import math
import os

import numpy as np
import torch


RECORDS_FILE = "records.jsonl"
LOCATIONS_FILE = "tensor_locations.jsonl"
TENSOR_FILE_TEMPLATE = "tensors-{file_id:03d}.data"
PLACEHOLDER_KEY = "__acc_tensor__"
UNSUPPORTED_PREFIX = "__acc_unsupported__:"


class JsonlWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "w")

    def write(self, item: dict):
        self._file.write(json.dumps(item, separators=(",", ":")) + "\n")

    def close(self):
        self._file.close()


def has_record_store(session_dir: str) -> bool:
    return os.path.exists(os.path.join(session_dir, RECORDS_FILE))


def load_info(json_path: str) -> dict:
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    save_id = os.path.splitext(os.path.basename(json_path))[0]
    base = os.path.dirname(json_path)
    for info in read_all_infos(base):
        if info.get("save_id") == save_id:
            return info
    raise FileNotFoundError(f"record info not found: {save_id}")


def load_data(pkl_path: str, dump_dir: str):
    save_id = os.path.splitext(os.path.basename(pkl_path))[0]
    info = None
    for item in read_all_infos(dump_dir):
        if item.get("save_id") == save_id:
            info = item
            break
    if info is None:
        raise FileNotFoundError(f"record data not found: {save_id}")

    locations = read_tensor_locations(dump_dir)
    args = _resolve_placeholders(info.get("args", []), locations, dump_dir)
    kwargs = _resolve_placeholders(info.get("kwargs", {}), locations, dump_dir)
    outputs = _resolve_placeholders(info.get("outputs", []), locations, dump_dir)
    return {"args": args, "kwargs": kwargs}, outputs


def read_all_infos(dump_dir: str):
    path = os.path.join(dump_dir, RECORDS_FILE)
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def read_tensor_locations(dump_dir: str):
    path = os.path.join(dump_dir, LOCATIONS_FILE)
    locations = {}
    if not os.path.exists(path):
        return locations
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            locations[item["id"]] = item
    return locations


def _resolve_placeholders(value, locations: dict, dump_dir: str):
    if isinstance(value, dict) and value.get(PLACEHOLDER_KEY):
        tensor_id = value["id"]
        if tensor_id not in locations:
            raise KeyError(f"tensor location missing: {tensor_id}")
        return _load_tensor(value, locations[tensor_id], dump_dir)
    if isinstance(value, dict):
        return {
            key: _resolve_placeholders(item, locations, dump_dir)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_placeholders(item, locations, dump_dir) for item in value]
    return value


def _load_tensor(placeholder: dict, location: dict, dump_dir: str):
    dtype_name = placeholder["dtype"]
    shape = placeholder["shape"]
    expected = math.prod(shape) * _element_size(dtype_name)
    if location["nbytes"] != expected:
        raise ValueError(
            f"tensor {placeholder['id']} size mismatch: "
            f"location={location['nbytes']} expected={expected}"
        )

    path = os.path.join(
        dump_dir, TENSOR_FILE_TEMPLATE.format(file_id=location["file"])
    )
    with open(path, "rb") as f:
        f.seek(location["offset"])
        raw = bytearray(f.read(location["nbytes"]))

    if placeholder["kind"] == "numpy":
        if location["nbytes"] == 0:
            return np.empty(shape, dtype=np.dtype(dtype_name))
        return np.frombuffer(raw, dtype=np.dtype(dtype_name)).reshape(shape)

    dtype = _torch_dtype(dtype_name)
    if location["nbytes"] == 0:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(raw, dtype=dtype).reshape(shape)

def _element_size(dtype_name: str) -> int:
    try:
        return torch.empty((), dtype=_torch_dtype(dtype_name)).element_size()
    except Exception:
        return np.dtype(dtype_name).itemsize


def _torch_dtype(dtype_name: str):
    return getattr(torch, dtype_name.replace("torch.", ""))
