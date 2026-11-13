import pytest

from src.experiments.sharding import select_shard


def test_shards_are_disjoint_and_cover_all_items():
    items = list(range(23))
    shards = [select_shard(items, index, 7) for index in range(7)]
    flattened = [item for shard in shards for item in shard]
    assert sorted(flattened) == items
    assert len(flattened) == len(set(flattened))


def test_shard_index_is_validated():
    with pytest.raises(ValueError):
        select_shard([1], 1, 1)
