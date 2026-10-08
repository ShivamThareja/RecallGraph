"""Memory evaluation framework — datasets, judges, metrics, runner, reports."""

from weavy.eval.dataset import EvalDataset, load_dataset
from weavy.eval.judges import KeywordJudge, LLMJudge, make_judge
from weavy.eval.report import EvalReport, render_text, write_json
from weavy.eval.runner import MemoryClient, run_eval

__all__ = [
    "EvalDataset",
    "EvalReport",
    "KeywordJudge",
    "LLMJudge",
    "MemoryClient",
    "load_dataset",
    "make_judge",
    "render_text",
    "run_eval",
    "write_json",
]
