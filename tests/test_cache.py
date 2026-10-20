import json

import numpy as np
import pytest

from halprobe.cache import (
    COMPLETE_MARKER,
    StageDir,
    append_jsonl,
    atomic_save_npy,
    atomic_write_json,
    read_jsonl,
    stage_key,
    write_jsonl,
)


def test_stage_key_is_stable_and_order_independent():
    a = stage_key({"x": 1, "y": [1, 2], "z": {"b": 2, "a": 1}})
    b = stage_key({"z": {"a": 1, "b": 2}, "y": [1, 2], "x": 1})
    assert a == b and len(a) == 12


def test_stage_key_changes_with_content():
    assert stage_key({"x": 1}) != stage_key({"x": 2})


def test_stage_dir_lifecycle(tmp_path):
    sd = StageDir(tmp_path, "data", "abc123")
    assert not sd.is_complete()
    work = sd.begin()
    assert work == tmp_path / "data" / "abc123.tmp"
    (work / "out.txt").write_text("hello", encoding="utf-8")
    assert not sd.is_complete()  # a .tmp dir is never a cache hit
    final = sd.commit()
    assert final == tmp_path / "data" / "abc123"
    assert sd.is_complete() and (final / COMPLETE_MARKER).exists()
    assert (final / "out.txt").read_text(encoding="utf-8") == "hello"
    assert not sd.tmp.exists()


def test_begin_keeps_partial_content_for_resume(tmp_path):
    sd = StageDir(tmp_path, "generate", "k")
    (sd.begin() / "partial.jsonl").write_text("{}\n", encoding="utf-8")
    assert (sd.begin() / "partial.jsonl").exists()


def test_commit_replaces_stale_final_without_marker(tmp_path):
    sd = StageDir(tmp_path, "data", "k")
    sd.final.mkdir(parents=True)
    (sd.final / "stale.txt").write_text("old", encoding="utf-8")
    sd.begin()
    sd.commit()
    assert not (sd.final / "stale.txt").exists() and sd.is_complete()


def test_atomic_json_and_numpy_types(tmp_path):
    path = tmp_path / "a.json"
    atomic_write_json(path, {"n": np.int64(3), "x": np.float32(0.5), "arr": np.array([1, 2])})
    assert json.loads(path.read_text(encoding="utf-8")) == {"n": 3, "x": 0.5, "arr": [1, 2]}
    assert not (tmp_path / "a.json.part").exists()


def test_atomic_save_npy_round_trip(tmp_path):
    path = tmp_path / "done.npy"
    atomic_save_npy(path, np.array([True, False]))
    atomic_save_npy(path, np.array([True, True]))
    assert np.load(path).tolist() == [True, True]


def test_jsonl_write_append_read(tmp_path):
    path = tmp_path / "r.jsonl"
    write_jsonl(path, [{"id": "a"}])
    append_jsonl(path, [{"id": "b"}, {"id": "c"}])
    assert [r["id"] for r in read_jsonl(path)] == ["a", "b", "c"]


def test_read_jsonl_partial_tail(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text('{"id": "a"}\n{"id": "b", "ans', encoding="utf-8")
    assert read_jsonl(path, tolerate_partial_tail=True) == [{"id": "a"}]
    with pytest.raises(ValueError, match="line 2"):
        read_jsonl(path)
