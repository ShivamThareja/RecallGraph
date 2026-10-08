"""
Agent harness runner — one loop engine shared by ingestion, query, and theme modes.
"""

import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import litellm
from falkordb import Graph

from weavy.config import settings
from weavy.harness import actions
from weavy.harness.tracing import RunTracer, finalize_trace, new_trace, record_turn
from weavy.services.provenance import annotate_citations
from weavy.store.traces import persist_trace

from weavy.models.traces import RunMode, RunTrace, ToolCall, Turn, TurnUsage

_MAX_PLAINTEXT_RETRIES = 1  # times to re-prompt before failing when model returns text instead of calling a tool
_MAX_TOOL_RETRIES = 3  # total tool-error nudges across the run before hard-failing


def _sanitize_message_for_trace(message: dict[str, Any]) -> dict[str, Any]:
    sanitized = deepcopy(message)
    sanitized.pop("tool_call_id", None)
    tool_calls = sanitized.get("tool_calls")
    if isinstance(tool_calls, list):
        sanitized["tool_calls"] = [
            {
                **{k: v for k, v in tc.items() if k != "id"},
                "function": {
                    k: v for k, v in tc.get("function", {}).items() if k != "id"
                },
            }
            for tc in tool_calls
            if isinstance(tc, dict)
        ]
    return sanitized


@lru_cache(maxsize=None)
def _get_context_limit(model: str) -> int | None:
    try:
        info = litellm.get_model_info(model)
    except Exception:
        return None
    return info.get("max_input_tokens") or None


def _finalize_failed_run(
    trace: RunTrace,
    tracer: RunTracer,
    graph: Graph,
    error: str,
    context_limit: int | None,
    turn: Turn | None = None,
) -> RunTrace:
    if turn is not None:
        record_turn(trace, turn)
    tracer.end_turn(None)
    finalize_trace(trace, "failed", error=error)
    persist_trace(graph, trace)
    tracer.finalize(trace, context_limit=context_limit)
    return trace


def _handle_tool_error(
    *,
    trace: RunTrace,
    tracer: RunTracer,
    turn: Turn,
    tool_name: str,
    args: dict[str, Any],
    error: str,
    called_at: datetime,
    tool_call_id: str,
    messages: list[dict[str, Any]],
    tool_error_retries: int,
    context_limit: int | None,
    graph: Graph,
) -> tuple[int, RunTrace | None]:
    tracer.record_tool_error(turn.turn_number, tool_name, args, error)
    turn.tool_calls.append(
        ToolCall(
            tool_name=tool_name,
            args=args,
            result=None,
            error=error,
            called_at=called_at,
        )
    )
    if tool_error_retries >= _MAX_TOOL_RETRIES:
        return tool_error_retries, _finalize_failed_run(
            trace, tracer, graph, error, context_limit, turn=turn
        )

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": f"Error: {error}",
        }
    )
    return tool_error_retries + 1, None


def _get_attr(obj: Any, key: str) -> int:
    """Read key from a dict or object attribute, returning 0 if absent or falsy."""
    val = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return val or 0


def _build_turn_usage(usage_raw: Any) -> TurnUsage:
    if not usage_raw:
        return TurnUsage()
    return TurnUsage(
        prompt_tokens=getattr(usage_raw, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage_raw, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage_raw, "total_tokens", 0) or 0,
        cached_tokens=_get_attr(
            getattr(usage_raw, "prompt_tokens_details", None), "cached_tokens"
        ),
        reasoning_tokens=_get_attr(
            getattr(usage_raw, "completion_tokens_details", None), "reasoning_tokens"
        ),
    )


def run(
    mode: RunMode,
    system_prompt: str,
    initial_messages: list[dict[str, Any]],
    allowed_actions: list[str],
    run_context: dict[str, Any],
    graph: Graph,
    session_id: str | None = None,
    parent_observation: Any = None,
    event_time: datetime | None = None,
) -> RunTrace:
    """
    Execute an agentic loop until the mode's completion tool is called.
    Returns the completed RunTrace.
    """
    input_summary = run_context.get("input_summary", "")
    trace = new_trace(mode, input_summary)
    trace.session_id = session_id
    tracer = RunTracer(
        trace.run_id,
        mode,
        input_summary,
        session_id=session_id,
        parent_observation=parent_observation,
    )
    ctx = actions.ActionContext(
        graph=graph, trace=trace, event_time=event_time, session_id=session_id
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *initial_messages,
    ]
    tool_definitions = actions.get_action_definitions(allowed_actions)
    turn_number = 0
    plaintext_retries = 0
    tool_error_retries = 0
    sanitized_cache: list[dict[str, Any]] = []

    context_limit = _get_context_limit(settings.LLM_MODEL)

    while True:
        turn_number += 1
        turn_ts = datetime.now(tz=timezone.utc)
        new_messages = messages[len(sanitized_cache) :]
        sanitized_cache.extend([_sanitize_message_for_trace(m) for m in new_messages])
        input_messages_snapshot = list(sanitized_cache)

        tracer.start_turn(turn_number, len(messages))

        # --- Call model ---
        tracer.prepare_llm_call()
        try:
            response = litellm.completion(
                model=settings.LLM_MODEL,
                messages=messages,
                tools=tool_definitions,
                tool_choice="auto",
                reasoning_effort=settings.REASONING_EFFORT,
            )
        except Exception as e:
            err = f"Model call failed: {e}"
            return _finalize_failed_run(trace, tracer, graph, err, context_limit)
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        # --- Build turn usage ---
        turn_usage = _build_turn_usage(getattr(response, "usage", None))

        reasoning = getattr(message, "reasoning_content", None)
        serialized_tool_calls = [
            {"name": tc.function.name, "args": tc.function.arguments}
            for tc in tool_calls
        ]

        tracer.record_llm_response(
            turn_number=turn_number,
            input_messages=input_messages_snapshot,
            text_content=message.content,
            reasoning_content=reasoning,
            tool_calls=serialized_tool_calls,
            usage=turn_usage,
        )

        turn = Turn(
            turn_number=turn_number,
            input_messages=input_messages_snapshot,
            reasoning_content=reasoning,
            text_content=message.content,
            usage=turn_usage,
            timestamp=turn_ts,
        )

        if not tool_calls:
            record_turn(trace, turn)
            if message.content and plaintext_retries < _MAX_PLAINTEXT_RETRIES:
                plaintext_retries += 1
                tracer.end_turn(message.content)
                messages.append({"role": "assistant", "content": message.content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You must call the completion tool to finish. "
                            "Do not respond with plain text — call complete "
                            "(or the appropriate completion tool for this mode) now."
                        ),
                    }
                )
                continue

            err = "Model stopped without calling a completion tool."
            return _finalize_failed_run(trace, tracer, graph, err, context_limit)

        # Append assistant message with tool_calls to conversation history.
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        # --- Process each tool call ---
        tool_errored = False
        for tc in tool_calls:
            tool_name = tc.function.name
            tool_call_id = tc.id
            called_at = datetime.now(tz=timezone.utc)

            if tool_errored:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": "Skipped: prior tool call in this batch failed.",
                    }
                )
                continue

            # Resolve tool
            entry = actions.ACTIONS.get(tool_name)
            if entry is None:
                err = f"Unknown tool '{tool_name}'."
                tool_error_retries, failed_trace = _handle_tool_error(
                    trace=trace,
                    tracer=tracer,
                    turn=turn,
                    tool_name=tool_name,
                    args={},
                    error=err,
                    called_at=called_at,
                    tool_call_id=tool_call_id,
                    messages=messages,
                    tool_error_retries=tool_error_retries,
                    context_limit=context_limit,
                    graph=graph,
                )
                if failed_trace is not None:
                    return failed_trace
                tool_errored = True
                continue

            # Parse args
            args_dict: dict[str, Any] = {}
            try:
                args_dict = json.loads(tc.function.arguments or "{}")
                params = entry.input_model(**args_dict)
            except Exception as e:
                err = f"Invalid arguments for '{tool_name}': {e}"
                tool_error_retries, failed_trace = _handle_tool_error(
                    trace=trace,
                    tracer=tracer,
                    turn=turn,
                    tool_name=tool_name,
                    args=args_dict,
                    error=err,
                    called_at=called_at,
                    tool_call_id=tool_call_id,
                    messages=messages,
                    tool_error_retries=tool_error_retries,
                    context_limit=context_limit,
                    graph=graph,
                )
                if failed_trace is not None:
                    return failed_trace
                tool_errored = True
                continue

            # Execute tool
            t0 = time.monotonic()
            try:
                result = entry.fn(params, ctx)
            except Exception as e:
                err = f"Tool '{tool_name}' raised: {e}"
                tool_error_retries, failed_trace = _handle_tool_error(
                    trace=trace,
                    tracer=tracer,
                    turn=turn,
                    tool_name=tool_name,
                    args=args_dict,
                    error=err,
                    called_at=called_at,
                    tool_call_id=tool_call_id,
                    messages=messages,
                    tool_error_retries=tool_error_retries,
                    context_limit=context_limit,
                    graph=graph,
                )
                if failed_trace is not None:
                    return failed_trace
                tool_errored = True
                continue

            duration_ms = (time.monotonic() - t0) * 1000
            result_str = (
                result.model_dump_json()
                if hasattr(result, "model_dump_json")
                else json.dumps(result)
            )

            tracer.record_tool_call(
                turn_number, tool_name, args_dict, result_str, duration_ms
            )
            messages.append(
                {"role": "tool", "tool_call_id": tool_call_id, "content": result_str}
            )
            turn.tool_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    args=args_dict,
                    result=result_str,
                    called_at=called_at,
                ),
            )

            if entry.is_completion:
                record_turn(trace, turn)
                trace.conversation = messages[1:]  # skip system message
                # Check cited sources/nodes against what the run actually
                # retrieved, before the trace is persisted.
                annotate_citations(trace)
                finalize_trace(trace, "completed")
                persist_trace(graph, trace)
                tracer.end_turn(message.content)
                tracer.finalize(trace, context_limit=context_limit)
                return trace

        # Turn complete — record and loop
        tracer.end_turn(message.content)
        record_turn(trace, turn)
