"""Eval dataset loading and validation. No FalkorDB or LLM required."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from weavy.eval.dataset import (
    DATASET_LOADERS,
    NATIVE_FORMAT,
    EvalConversation,
    EvalDataset,
    list_builtin_datasets,
    load_dataset,
)


def _minimal(**overrides) -> dict:
    data = {
        "format": NATIVE_FORMAT,
        "name": "mini",
        "conversations": [
            {
                "id": "c1",
                "sessions": [
                    {"id": "s1", "timestamp": "2024-01-01T10:00:00+00:00", "text": "hi"}
                ],
                "questions": [
                    {
                        "id": "q1",
                        "question": "?",
                        "answer": "a",
                        "evidence_sessions": ["s1"],
                    }
                ],
            }
        ],
    }
    data.update(overrides)
    return data


def _write(tmp_path, data) -> str:
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_builtin_smoke_dataset_loads_and_is_consistent() -> None:
    assert "smoke" in list_builtin_datasets()
    ds = load_dataset("smoke")
    assert ds.name == "smoke"
    assert ds.question_count == 14
    assert ds.source_sha256 and len(ds.source_sha256) == 64
    categories = {q.category for c in ds.conversations for q in c.questions}
    assert {
        "single-hop",
        "multi-hop",
        "temporal",
        "knowledge-update",
        "abstention",
    } <= categories
    # The dataset must be honest about not being a benchmark.
    assert "NOT a benchmark" in ds.description


def test_load_from_path_records_hash(tmp_path) -> None:
    ds = load_dataset(_write(tmp_path, _minimal()))
    assert ds.question_count == 1
    assert ds.source_sha256 is not None


def test_unknown_dataset_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset("definitely-not-a-dataset")


def test_unknown_format_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown dataset format"):
        load_dataset(_write(tmp_path, _minimal()), fmt="locomo")


def test_format_mismatch_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="expected format"):
        load_dataset(_write(tmp_path, _minimal(format="other/v9")))


def test_naive_timestamp_rejected(tmp_path) -> None:
    data = _minimal()
    data["conversations"][0]["sessions"][0]["timestamp"] = "2024-01-01T10:00:00"
    with pytest.raises(ValidationError):
        load_dataset(_write(tmp_path, data))


def test_unknown_evidence_session_rejected(tmp_path) -> None:
    data = _minimal()
    data["conversations"][0]["questions"][0]["evidence_sessions"] = ["nope"]
    with pytest.raises(ValidationError, match="unknown evidence_sessions"):
        load_dataset(_write(tmp_path, data))


def test_duplicate_question_ids_rejected(tmp_path) -> None:
    data = _minimal()
    conv = data["conversations"][0]
    data["conversations"].append({**conv, "id": "c2"})
    with pytest.raises(ValidationError, match="duplicate question ids"):
        load_dataset(_write(tmp_path, data))


def test_abstain_question_cannot_have_evidence(tmp_path) -> None:
    data = _minimal()
    data["conversations"][0]["questions"][0]["abstain"] = True
    with pytest.raises(ValidationError, match="abstention"):
        load_dataset(_write(tmp_path, data))


def test_query_time_resolution_and_utc_normalization() -> None:
    conv = EvalConversation.model_validate(
        {
            "id": "c",
            "sessions": [
                {"id": "a", "timestamp": "2024-01-02T10:00:00+00:00", "text": "x"},
                {"id": "b", "timestamp": "2024-01-01T10:00:00+00:00", "text": "y"},
            ],
            "questions": [
                {"id": "q1", "question": "?", "answer": "a"},
                {
                    "id": "q2",
                    "question": "?",
                    "answer": "a",
                    "query_time": "2024-02-01T12:00:00+05:30",
                },
            ],
        }
    )
    assert [s.id for s in conv.sessions_in_order()] == ["b", "a"]
    q1, q2 = conv.questions
    # No overrides → latest session time.
    assert conv.resolve_query_time(q1) == datetime(2024, 1, 2, 10, tzinfo=timezone.utc)
    # Offsets are normalized to UTC because Weavy renders times with a 'Z'.
    resolved = conv.resolve_query_time(q2)
    assert resolved.tzinfo == timezone.utc
    assert resolved == datetime(2024, 2, 1, 6, 30, tzinfo=timezone.utc)


def test_loader_registry_is_extensible(tmp_path, monkeypatch) -> None:
    def fake_loader(data: dict) -> EvalDataset:
        return EvalDataset.model_validate(
            {"name": data["title"], "conversations": _minimal()["conversations"]}
        )

    monkeypatch.setitem(DATASET_LOADERS, "fake/v1", fake_loader)
    path = _write(tmp_path, {"title": "converted"})
    ds = load_dataset(path, fmt="fake/v1")
    assert ds.name == "converted"
