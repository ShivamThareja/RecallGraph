"""Time normalization: every datetime Weavy stores or compares is UTC-aware.

Naive datetimes carry no offset, so Weavy interprets them as UTC — the same
convention the prompts use when they render times with a trailing ``Z``.
Aware datetimes are converted to UTC. Comparing a mix of the two therefore
never raises, and ISO strings written by Weavy sort chronologically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from pydantic import AfterValidator


def to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def to_utc_opt(value: datetime | None) -> datetime | None:
    return None if value is None else to_utc(value)


# Pydantic field type: accepts naive or aware input, always yields aware UTC.
UTCDatetime = Annotated[datetime, AfterValidator(to_utc)]


def parse_utc(value: str) -> datetime:
    """Parse an ISO-8601 string (naive treated as UTC) into an aware UTC datetime."""
    return to_utc(datetime.fromisoformat(value))


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def normalize_range(
    bounds: list[datetime | None] | None,
) -> tuple[datetime | None, datetime | None] | None:
    """Validate and normalize a ``[start, end]`` range; either bound may be None.

    Returns None when there is no constraint at all (no range, or both bounds
    open). Raises ValueError for a malformed or inverted range.
    """
    if bounds is None:
        return None
    if len(bounds) != 2:
        raise ValueError("time range must be [start, end]; use null for an open bound")
    start, end = to_utc_opt(bounds[0]), to_utc_opt(bounds[1])
    if start is None and end is None:
        return None
    if start is not None and end is not None and start > end:
        raise ValueError("time range start is after its end")
    return start, end


def in_range(when: datetime, start: datetime | None, end: datetime | None) -> bool:
    when = to_utc(when)
    if start is not None and when < start:
        return False
    if end is not None and when > end:
        return False
    return True
