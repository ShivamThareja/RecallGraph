"""Evaluation dataset model, validation, and format registry.

The native format (``weavy-eval/v1``) is a JSON document::

    {
      "format": "weavy-eval/v1",
      "name": "smoke", "version": "1", "description": "...",
      "conversations": [
        {
          "id": "c1",
          "context": "optional caller context passed to every ingestion",
          "query_time": "2024-03-20T12:00:00+00:00",   # default for its questions
          "sessions":  [{"id": "c1s1", "timestamp": "...", "text": "..."}],
          "questions": [{
            "id": "c1q1", "question": "...", "answer": "...",
            "category": "single-hop",
            "answer_keywords":   ["Lisbon", "June|jun"],  # all groups required
            "evidence_sessions": ["c1s1"],                # local session ids
            "evidence_keywords": ["Lisbon"],              # facts that must surface
            "abstain": false,
            "query_time": null                            # overrides the conversation's
          }]
        }
      ]
    }

Session ids are *local* to the dataset; the runner maps them to the ``s:N`` ids
Weavy mints at ingestion. A keyword entry may list alternatives separated by
``|`` — the group matches when any alternative matches.

Other benchmark formats (LoCoMo, LongMemEval, ...) plug in by registering a
loader in ``DATASET_LOADERS`` that converts their file into ``EvalDataset``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import AwareDatetime, BaseModel, Field, field_validator, model_validator

NATIVE_FORMAT = "weavy-eval/v1"
BUILTIN_DIR = Path(__file__).parent / "datasets"


class EvalSession(BaseModel):
    id: str
    timestamp: AwareDatetime
    text: str

    @field_validator("text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("session text must be non-empty")
        return v


class EvalQuestion(BaseModel):
    id: str
    question: str
    answer: str
    category: str = "uncategorized"
    answer_keywords: list[str] = Field(default_factory=list)
    evidence_sessions: list[str] = Field(default_factory=list)
    evidence_keywords: list[str] = Field(default_factory=list)
    abstain: bool = False
    query_time: AwareDatetime | None = None

    @model_validator(mode="after")
    def _abstain_has_no_evidence(self) -> EvalQuestion:
        if self.abstain and self.evidence_sessions:
            raise ValueError(
                f"question {self.id!r}: abstention questions cannot have evidence_sessions"
            )
        return self


class EvalConversation(BaseModel):
    id: str
    context: str | None = None
    query_time: AwareDatetime | None = None
    sessions: list[EvalSession]
    questions: list[EvalQuestion]

    @model_validator(mode="after")
    def _references_resolve(self) -> EvalConversation:
        session_ids = [s.id for s in self.sessions]
        if len(set(session_ids)) != len(session_ids):
            raise ValueError(f"conversation {self.id!r}: duplicate session ids")
        known = set(session_ids)
        for q in self.questions:
            missing = [sid for sid in q.evidence_sessions if sid not in known]
            if missing:
                raise ValueError(
                    f"question {q.id!r}: unknown evidence_sessions {missing}"
                )
        return self

    def sessions_in_order(self) -> list[EvalSession]:
        return sorted(self.sessions, key=lambda s: s.timestamp)

    def resolve_query_time(self, question: EvalQuestion) -> datetime:
        """Question override → conversation default → latest session time (UTC)."""
        if question.query_time is not None:
            chosen = question.query_time
        elif self.query_time is not None:
            chosen = self.query_time
        else:
            chosen = max(s.timestamp for s in self.sessions)
        return chosen.astimezone(timezone.utc)


class EvalDataset(BaseModel):
    name: str
    version: str = "1"
    description: str = ""
    conversations: list[EvalConversation]
    # Filled by load_dataset: sha256 of the source file, for reproducibility.
    source_sha256: str | None = None

    @model_validator(mode="after")
    def _unique_ids(self) -> EvalDataset:
        conv_ids = [c.id for c in self.conversations]
        if len(set(conv_ids)) != len(conv_ids):
            raise ValueError("duplicate conversation ids")
        q_ids = [q.id for c in self.conversations for q in c.questions]
        if len(set(q_ids)) != len(q_ids):
            raise ValueError("duplicate question ids")
        return self

    @property
    def question_count(self) -> int:
        return sum(len(c.questions) for c in self.conversations)


def _load_native(data: dict) -> EvalDataset:
    fmt = data.get("format", NATIVE_FORMAT)
    if fmt != NATIVE_FORMAT:
        raise ValueError(f"expected format {NATIVE_FORMAT!r}, got {fmt!r}")
    payload = {k: v for k, v in data.items() if k != "format"}
    return EvalDataset.model_validate(payload)


# Format name → loader converting parsed JSON into an EvalDataset.
DATASET_LOADERS: dict[str, Callable[[dict], EvalDataset]] = {
    NATIVE_FORMAT: _load_native,
}


def list_builtin_datasets() -> list[str]:
    return sorted(p.stem for p in BUILTIN_DIR.glob("*.json"))


def resolve_dataset_path(name_or_path: str) -> Path:
    """A built-in dataset name (e.g. ``smoke``) or a filesystem path."""
    builtin = BUILTIN_DIR / f"{name_or_path}.json"
    if builtin.is_file():
        return builtin
    path = Path(name_or_path)
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"dataset {name_or_path!r} is neither a built-in "
        f"({', '.join(list_builtin_datasets()) or 'none'}) nor an existing file"
    )


def load_dataset(name_or_path: str, fmt: str = NATIVE_FORMAT) -> EvalDataset:
    if fmt not in DATASET_LOADERS:
        raise ValueError(
            f"unknown dataset format {fmt!r}; known: {', '.join(sorted(DATASET_LOADERS))}"
        )
    path = resolve_dataset_path(name_or_path)
    raw = path.read_bytes()
    dataset = DATASET_LOADERS[fmt](json.loads(raw))
    dataset.source_sha256 = hashlib.sha256(raw).hexdigest()
    return dataset
