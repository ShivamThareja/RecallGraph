"""Eval judges and cost estimation. No FalkorDB or real LLM required."""

from unittest.mock import MagicMock, patch

import pytest

from weavy.eval.dataset import EvalQuestion
from weavy.eval.judges import (
    KeywordJudge,
    LLMJudge,
    is_abstention,
    keyword_group_matches,
    make_judge,
    normalize,
)
from weavy.services import cost


def _q(**kw) -> EvalQuestion:
    base = {"id": "q", "question": "?", "answer": "Biscuit"}
    base.update(kw)
    return EvalQuestion.model_validate(base)


def test_normalize_strips_case_accents_punctuation() -> None:
    assert normalize("  Tomás, the  ENGINEER! ") == "tomas the engineer"


@pytest.mark.parametrize(
    "group,text,expected",
    [
        ("Biscuit", "Your dog is called Biscuit.", True),
        ("biscuit", "BISCUIT", True),
        ("Bisc", "Biscuit", False),  # whole words only
        ("4 March|2024-03-04", "It was on 2024-03-04.", True),
        ("4 March|2024-03-04", "It was on 14 March.", False),
        ("Tomás", "Tomas went on leave", True),
    ],
)
def test_keyword_group_matches(group, text, expected) -> None:
    assert keyword_group_matches(group, text) is expected


def test_keyword_judge_requires_all_groups() -> None:
    judge = KeywordJudge()
    q = _q(answer="Lena in Berlin", answer_keywords=["Lena", "Berlin"])
    assert judge.judge(q, "Your sister Lena lives in Berlin.").correct
    verdict = judge.judge(q, "Your sister Lena helped.")
    assert not verdict.correct
    assert "Berlin" in verdict.reason


def test_keyword_judge_falls_back_to_gold_answer() -> None:
    assert KeywordJudge().judge(_q(), "It's Biscuit").correct


def test_keyword_judge_empty_answer_is_wrong() -> None:
    assert not KeywordJudge().judge(_q(), "   ").correct


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("The graph has no record of this.", True),
        ("I couldn't find it — there's no information about a cat.", True),
        ("A cat was never mentioned.", True),
        ("Your cat is called Biscuit.", False),
    ],
)
def test_abstention_detection(answer, expected) -> None:
    assert is_abstention(answer) is expected


def test_keyword_judge_abstention_question() -> None:
    q = _q(answer="no record", abstain=True)
    judge = KeywordJudge()
    assert judge.judge(q, "The graph has no record of this.").correct
    assert not judge.judge(q, "Your cat is Whiskers.").correct


def _llm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 10
    return resp


def test_llm_judge_parses_verdict_and_usage() -> None:
    reply = 'Sure: {"verdict": "CORRECT", "reason": "matches"}'
    with patch("litellm.completion", return_value=_llm_response(reply)) as completion:
        verdict = LLMJudge("some/model").judge(_q(), "Biscuit")
    assert verdict.correct
    assert verdict.reason == "matches"
    assert verdict.prompt_tokens == 100 and verdict.completion_tokens == 10
    prompt = completion.call_args.kwargs["messages"][0]["content"]
    assert "Gold answer: Biscuit" in prompt


def test_llm_judge_abstention_uses_abstain_rule() -> None:
    reply = '{"verdict": "WRONG", "reason": "asserted an answer"}'
    with patch("litellm.completion", return_value=_llm_response(reply)) as completion:
        verdict = LLMJudge("m").judge(_q(abstain=True, answer="no record"), "Whiskers")
    assert not verdict.correct
    assert "NO record" in completion.call_args.kwargs["messages"][0]["content"]


def test_llm_judge_unparseable_reply_is_wrong_not_guessed() -> None:
    with patch("litellm.completion", return_value=_llm_response("looks right to me")):
        verdict = LLMJudge("m").judge(_q(), "Biscuit")
    assert not verdict.correct
    assert "unparseable" in verdict.reason


def test_make_judge() -> None:
    assert make_judge("keyword").name == "keyword"
    assert make_judge("llm", "m").name == "llm"
    with pytest.raises(ValueError):
        make_judge("llm")
    with pytest.raises(ValueError):
        make_judge("vibes")


def test_estimate_cost_known_and_unknown_models(monkeypatch) -> None:
    cost.model_cost_per_token.cache_clear()
    monkeypatch.setattr(
        cost.litellm,
        "model_cost",
        {"priced": {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}},
    )
    try:
        assert cost.estimate_cost("priced", 1000, 500) == pytest.approx(0.002)
        # Unknown price is None, never a silent $0.
        assert cost.estimate_cost("unpriced", 1000, 500) is None
    finally:
        cost.model_cost_per_token.cache_clear()
