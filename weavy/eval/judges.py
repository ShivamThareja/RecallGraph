"""Answer judges.

``KeywordJudge`` is deterministic and offline: every required keyword group must
appear in the normalized answer. It is strict about wording and lenient about
extra text, so it is a reproducible floor, not a semantic grader.

``LLMJudge`` asks a model whether the answer matches the gold answer. It is
closer to how LoCoMo/LongMemEval-style benchmarks grade, but it is
non-deterministic and costs tokens; its usage is reported separately from the
system under test.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Protocol

import litellm
from pydantic import BaseModel

from weavy.eval.dataset import EvalQuestion
from weavy.services.cost import estimate_cost

# Phrases that signal the system declined to answer for lack of evidence. The
# query prompt instructs the agent to say "The graph has no record of this."
ABSTENTION_PATTERNS = [
    r"\bno record\b",
    r"\bno (?:information|evidence|mention|data)\b",
    r"\b(?:not|never) (?:been )?(?:mentioned|recorded|stored|found)\b",
    r"\bdoes(?:n't| not) (?:contain|have|mention|say)\b",
    r"\b(?:cannot|can't|unable to) (?:find|answer|determine)\b",
    r"\bno .{0,40}\b(?:found|recorded|stored)\b",
]
_ABSTENTION_RE = re.compile("|".join(ABSTENTION_PATTERNS), re.IGNORECASE)


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def keyword_group_matches(group: str, text: str) -> bool:
    """True when any ``|``-separated alternative appears as whole words in text."""
    haystack = f" {normalize(text)} "
    for alternative in group.split("|"):
        needle = normalize(alternative)
        if needle and f" {needle} " in haystack:
            return True
    return False


def is_abstention(answer: str) -> bool:
    return bool(_ABSTENTION_RE.search(answer))


class Verdict(BaseModel):
    correct: bool
    reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = 0.0


class Judge(Protocol):
    name: str

    def judge(self, question: EvalQuestion, answer: str) -> Verdict: ...


class KeywordJudge:
    name = "keyword"

    def judge(self, question: EvalQuestion, answer: str) -> Verdict:
        if not answer.strip():
            return Verdict(correct=False, reason="empty answer")
        if question.abstain:
            ok = is_abstention(answer)
            return Verdict(
                correct=ok,
                reason="abstained as expected"
                if ok
                else "answered instead of abstaining",
            )
        groups = question.answer_keywords or [question.answer]
        missing = [g for g in groups if not keyword_group_matches(g, answer)]
        if missing:
            return Verdict(correct=False, reason=f"missing keywords: {missing}")
        return Verdict(correct=True, reason="all keyword groups matched")


_LLM_JUDGE_PROMPT = """You are grading an answer produced by a memory system.

Question: {question}
Gold answer: {gold}
System answer: {answer}

{rule}

Reply with JSON only: {{"verdict": "CORRECT" | "WRONG", "reason": "<one sentence>"}}"""

_ANSWER_RULE = (
    "Mark CORRECT if the system answer contains the information in the gold "
    "answer (paraphrases, equivalent dates, and extra correct detail are fine). "
    "Mark WRONG if it is missing, contradicts, or only hedges about the gold answer."
)
_ABSTAIN_RULE = (
    "The gold answer is that the memory has NO record of this. Mark CORRECT "
    "only if the system says it has no record / cannot find it. Mark WRONG if "
    "it asserts any answer."
)


class LLMJudge:
    name = "llm"

    def __init__(self, model: str) -> None:
        self.model = model

    def judge(self, question: EvalQuestion, answer: str) -> Verdict:
        if not answer.strip():
            return Verdict(correct=False, reason="empty answer")
        prompt = _LLM_JUDGE_PROMPT.format(
            question=question.question,
            gold=question.answer,
            answer=answer,
            rule=_ABSTAIN_RULE if question.abstain else _ANSWER_RULE,
        )
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        verdict, reason = _parse_verdict(content)
        return Verdict(
            correct=verdict,
            reason=reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=estimate_cost(self.model, prompt_tokens, completion_tokens),
        )


def _parse_verdict(content: str) -> tuple[bool, str]:
    """Parse the judge reply; an unparseable reply is graded WRONG, not guessed."""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            verdict = str(data.get("verdict", "")).strip().upper()
            if verdict in ("CORRECT", "WRONG"):
                return verdict == "CORRECT", str(data.get("reason", ""))
        except json.JSONDecodeError:
            pass
    return False, f"unparseable judge reply: {content[:200]!r}"


def make_judge(name: str, model: str | None = None) -> Judge:
    if name == "keyword":
        return KeywordJudge()
    if name == "llm":
        if not model:
            raise ValueError("the llm judge requires a judge model")
        return LLMJudge(model)
    raise ValueError(f"unknown judge {name!r}; expected 'keyword' or 'llm'")
