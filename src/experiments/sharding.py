from __future__ import annotations


def select_shard(items: list, shard_index: int, shard_count: int) -> list:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    start = len(items) * shard_index // shard_count
    stop = len(items) * (shard_index + 1) // shard_count
    return items[start:stop]
