"""
Run trace helpers — optional Langfuse-backed tracer and in-memory RunTrace state management.

RunTracer: wraps Langfuse observations for a single agent run when Langfuse is
configured. Creates a root span with nested turn spans and generation/tool child
observations. When Langfuse is not configured, only console output is produced.

Langfuse observation hierarchy (standalone run):
    root-span (ingestion-run / query-run / theme-run)
    └── turn-1 span
        ├── llm generation  (created BEFORE litellm call for accurate timing)
        ├── tool:search_graph span
        └── tool:create_node span
    └── turn-2 span
        ├── llm generation
        └── tool:complete span

Langfuse observation hierarchy (interactive chat session via ChatSessionTracer):
    chat-session root-span
    └── query-run (message 1)
        └── turn-1 span ...
    └── query-run (message 2)
        └── turn-1 span ...

The in-memory RunTrace (new_trace / record_turn / finalize_trace) remains the
authoritative in-process state container for touched_nodes, completion_payload,
status, and turn records. It is NOT persisted to disk.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from weavy.config import settings
from weavy.models.traces import RunMode, RunTrace, TurnUsage, graph_delta
from weavy.services.cost import model_cost_per_token


def _model_cost_per_token(model: str) -> tuple[float, float]:
    """Return (input_cost_per_token, output_cost_per_token) for the model, or (0, 0) if unknown."""
    return model_cost_per_token(model) or (0.0, 0.0)


def _truncate(s: str, n: int = 120) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _coerce_langfuse_trace_id(trace_id: str) -> str | None:
    """Return a Langfuse-safe 32-char lowercase hex trace id when possible."""
    if len(trace_id) == 32 and all(c in "0123456789abcdef" for c in trace_id):
        return trace_id
    try:
        return uuid.UUID(trace_id).hex
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Langfuse availability check
# ---------------------------------------------------------------------------


def _langfuse_enabled() -> bool:
    return bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY)


# ---------------------------------------------------------------------------
# Optional Langfuse-backed RunTracer
# ---------------------------------------------------------------------------


class ChatSessionTracer:
    """Long-lived tracer for an interactive chat session.

    Creates one root Langfuse span for the entire session when Langfuse is
    configured. Pass the root observation into RunTracer so each per-message
    run is a child span rather than an independent trace.

    Usage (in run_chat_repl):
        session = ChatSessionTracer(chat_id)
        # per message:
        RunTracer(..., parent_observation=session.root)
        # at end:
        session.finalize(message_count)
    """

    def __init__(self, chat_id: str) -> None:
        self._lf = None
        self.root = None
        if _langfuse_enabled():
            from weavy.langfuse_client import get_langfuse

            self._lf = get_langfuse()
            self.root = self._lf.start_observation(
                name="chat-session",
                as_type="span",
                input={"chat_id": chat_id},
                metadata={"chat_id": chat_id},
            )

    def finalize(self, message_count: int) -> None:
        if self.root is not None:
            self.root.update(metadata={"total_messages": message_count})
            self.root.end()
        if self._lf is not None:
            self._lf.flush()


class RunTracer:
    """One Langfuse span per agent run (when Langfuse is configured).

    When parent_observation is given the run span is created as a child of that
    observation (used for chat sessions so all turns share one root trace).
    When parent_observation is None a new root trace is created.
    When Langfuse is not configured, only console output is produced.

    Lifecycle (called from runner.py):
        tracer = RunTracer(run_id, mode, input_summary)
        tracer.start_turn(n, message_count)
        tracer.prepare_llm_call()         # before litellm.completion()
        # ... call litellm.completion() ...
        tracer.record_llm_response(...)   # after litellm.completion()
        tracer.record_tool_call(...)      # per tool
        tracer.end_turn(text_content)
        # ... repeat for each turn ...
        tracer.finalize(trace)
    """

    def __init__(
        self,
        run_id: str,
        mode: str,
        input_summary: str,
        session_id: str | None = None,
        parent_observation: Any = None,
    ) -> None:
        self._lf = None
        self._root = None
        self._owns_root = parent_observation is None
        self._trace_id = run_id
        self._current_turn_span: Any = None
        self._current_generation: Any = None

        if _langfuse_enabled():
            from weavy.langfuse_client import get_langfuse

            self._lf = get_langfuse()
            start_kwargs: dict[str, Any] = {
                "name": f"{mode}-run",
                "as_type": "span",
                "input": input_summary,
                "metadata": {"mode": mode, "session_id": session_id},
            }
            if parent_observation is not None:
                self._root = parent_observation.start_observation(**start_kwargs)
            else:
                trace_id = _coerce_langfuse_trace_id(run_id)
                if trace_id is not None:
                    start_kwargs["trace_context"] = {"trace_id": trace_id}
                self._root = self._lf.start_observation(**start_kwargs)

        print(f"[run_start] mode={mode} | {input_summary}")

    def get_trace_id(self) -> str:
        return self._trace_id

    # ---- Turn lifecycle ----

    def start_turn(self, turn_number: int, message_count: int) -> None:
        if self._root is not None:
            self._current_turn_span = self._root.start_observation(
                name=f"turn-{turn_number}",
                as_type="span",
                input={"message_count": message_count},
            )

    def end_turn(self, text_content: str | None) -> None:
        if self._current_generation is not None:
            self._current_generation.end()
            self._current_generation = None
        if self._current_turn_span is not None:
            if text_content is not None:
                self._current_turn_span.update(output=text_content)
            self._current_turn_span.end()
            self._current_turn_span = None

    def _close_open_spans(self) -> None:
        """Safety net: close any spans left open on error paths."""
        self.end_turn(None)

    # ---- LLM call (created before the call for accurate timing) ----

    def prepare_llm_call(self) -> None:
        """Open a generation span immediately before litellm.completion()."""
        if self._current_turn_span is None:
            return
        self._current_generation = self._current_turn_span.start_observation(
            name="llm",
            as_type="generation",
            model=settings.LLM_MODEL,
            model_parameters={"reasoning_effort": settings.REASONING_EFFORT},
        )

    def record_llm_response(
        self,
        turn_number: int,
        input_messages: list[dict[str, Any]],
        text_content: str | None,
        reasoning_content: str | None,
        tool_calls: list[dict[str, Any]],
        usage: TurnUsage,
    ) -> None:
        """Update the open generation span with response data and end it."""
        if self._current_generation is not None:
            output: dict[str, Any] = {}
            if text_content:
                output["content"] = text_content
            if reasoning_content:
                output["reasoning"] = reasoning_content
            if tool_calls:
                output["tool_calls"] = tool_calls

            input_cpt, output_cpt = _model_cost_per_token(settings.LLM_MODEL)
            input_cost = input_cpt * usage.prompt_tokens
            output_cost = output_cpt * usage.completion_tokens

            usage_info: dict[str, int] = {
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "total": usage.total_tokens,
                "cache_read_input_tokens": usage.cached_tokens,
            }

            self._current_generation.update(
                input=input_messages,
                output=output,
                usage_details=usage_info,
                cost_details={
                    "input": input_cost,
                    "output": output_cost,
                    "total": input_cost + output_cost,
                },
                metadata={
                    "reasoning_tokens": usage.reasoning_tokens,
                    "cached_tokens": usage.cached_tokens,
                },
            )
            self._current_generation.end()
            self._current_generation = None

        # Console output
        cache_str = f" cached={usage.cached_tokens}" if usage.cached_tokens else ""
        token_info = (
            f"prompt={usage.prompt_tokens}{cache_str}"
            f" completion={usage.completion_tokens}"
            f" reasoning={usage.reasoning_tokens} total={usage.total_tokens}"
        )
        if reasoning_content:
            print(f"\n[T{turn_number}] 💭 {_truncate(reasoning_content)}")
        elif text_content:
            print(f"\n[T{turn_number}] 💬 {_truncate(text_content)}")
        else:
            print(f"\n[T{turn_number}] ▶ ({len(tool_calls)} tool call(s))")
        print(f"       tokens: {token_info}")

    # ---- Tool calls ----

    def record_tool_call(
        self,
        turn_number: int,
        name: str,
        args: dict[str, Any],
        result: str,
        duration_ms: float,
    ) -> None:
        if self._current_turn_span is not None:
            span = self._current_turn_span.start_observation(
                name=f"tool:{name}",
                as_type="tool",
                input=args,
                metadata={"duration_ms": round(duration_ms, 1)},
            )
            span.update(output=result)
            span.end()
        print(
            f"[T{turn_number}] ← {name} ({duration_ms:.0f}ms): {_truncate(result, 80)}"
        )

    def record_tool_error(
        self,
        turn_number: int,
        name: str,
        args: dict[str, Any],
        error: str,
    ) -> None:
        if self._current_turn_span is not None:
            span = self._current_turn_span.start_observation(
                name=f"tool:{name}",
                as_type="tool",
                input=args,
                level="ERROR",
                status_message=error,
            )
            span.end()
        print(f"[T{turn_number}] ✗ {name}: {error}")

    # ---- Run completion ----

    def finalize(self, trace: RunTrace, *, context_limit: int | None = None) -> None:
        self._close_open_spans()

        delta = graph_delta(trace.touched_nodes, trace.touched_edges)
        total_turns = len(trace.turns)

        if self._root is not None:
            self._root.set_trace_io(output=trace.completion_payload)
            self._root.update(
                usage_details={
                    "input": trace.total_usage.prompt_tokens,
                    "output": trace.total_usage.completion_tokens,
                    "total": trace.total_usage.total_tokens,
                    "cache_read_input_tokens": trace.total_usage.cached_tokens,
                },
                metadata={
                    "status": trace.status,
                    "total_turns": total_turns,
                    "error": trace.error,
                    "reasoning_tokens": trace.total_usage.reasoning_tokens,
                    **delta,
                },
            )
            self._root.end()
        if self._owns_root and self._lf is not None:
            self._lf.flush()

        # Console summary
        status_icon = "✓" if trace.status == "completed" else "✗"
        prompt = trace.total_usage.prompt_tokens
        cached = trace.total_usage.cached_tokens
        if context_limit:
            pct = prompt / context_limit * 100
            ctx_str = f"{prompt:,} / {context_limit:,} ({pct:.1f}%)"
        else:
            ctx_str = f"{prompt:,}"
        cache_str = f" | cached: {cached:,}" if cached else ""
        print(
            f"\n[{status_icon}] {trace.status} | {total_turns} turn(s)"
            f"\n    context: {ctx_str}{cache_str}"
            f" | output: {trace.total_usage.completion_tokens:,}"
            f" | reasoning: {trace.total_usage.reasoning_tokens:,}"
        )
        if delta:
            pairs = "  ".join(f"{k}={v}" for k, v in delta.items())
            print(f"    graph: {pairs}")
        if trace.error:
            print(f"    error: {trace.error}")


# ---------------------------------------------------------------------------
# In-memory RunTrace helpers (state container — not persisted)
# ---------------------------------------------------------------------------


def new_trace(
    mode: RunMode,
    input_summary: str,
) -> RunTrace:
    return RunTrace(
        mode=mode,
        started_at=datetime.now(tz=timezone.utc),
        input_summary=input_summary,
        status="running",
    )


def record_turn(trace: RunTrace, turn: Any) -> None:
    trace.turns.append(turn)
    trace.total_usage.prompt_tokens += turn.usage.prompt_tokens
    trace.total_usage.completion_tokens += turn.usage.completion_tokens
    trace.total_usage.reasoning_tokens += turn.usage.reasoning_tokens
    trace.total_usage.cached_tokens += turn.usage.cached_tokens
    trace.total_usage.total_tokens += turn.usage.total_tokens


def finalize_trace(
    trace: RunTrace,
    status: Literal["completed", "failed"],
    completion_payload: dict | None = None,
    error: str | None = None,
) -> RunTrace:
    trace.status = status
    trace.ended_at = datetime.now(tz=timezone.utc)
    if completion_payload is not None:
        trace.completion_payload = completion_payload
    if error is not None:
        trace.error = error
    return trace
