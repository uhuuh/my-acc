"""Tensor byte/hash helpers shared by dump stages."""

import os

import torch


def tensor_bytes(tensor: torch.Tensor):
    tensor = tensor.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.numel() == 0:
        return memoryview(b"")
    return memoryview(tensor.numpy()).cast("B")


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def hash_tensor(tensor: torch.Tensor) -> str:
    try:
        import xxhash

        return xxhash.xxh3_128_hexdigest(tensor_bytes(tensor))
    except ImportError:
        import hashlib

        return hashlib.md5(bytes(tensor_bytes(tensor))).hexdigest()


def write_all(fd: int, data):
    offset = 0
    total = len(data)
    while offset < total:
        written = os.write(fd, data[offset:])
        if written == 0:
            raise OSError("os.write returned 0 bytes")
        offset += written
