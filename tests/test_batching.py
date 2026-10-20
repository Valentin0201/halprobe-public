import pytest
import torch

from halprobe.batching import iter_with_backoff


def test_backoff_halves_batch_on_oom():
    calls = []

    def fn(batch):
        calls.append(len(batch))
        if len(batch) > 2:
            raise torch.OutOfMemoryError("fake")
        return [x * 2 for x in batch]

    events = []
    out = list(iter_with_backoff(list(range(5)), 8, fn, on_backoff=events.append))
    assert [list(b) for b, _ in out] == [[0, 1], [2, 3], [4]]
    assert [r for _, r in out] == [[0, 2], [4, 6], [8]]
    assert events == [4, 2]
    assert calls == [5, 4, 2, 2, 1]


def test_backoff_reraises_at_batch_size_one():
    def fn(batch):
        raise torch.OutOfMemoryError("fake")

    with pytest.raises(torch.OutOfMemoryError):
        list(iter_with_backoff([1, 2], 2, fn))
