from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from falkordb import Graph

from weavy.models.themes import Theme
from weavy.store import themes as store_themes
from weavy.store.system import SystemState


@lru_cache(maxsize=None)
def _load_prompt_template(name: str) -> str:
    prompts_dir = Path(__file__).parent.parent / "prompts"
    for suffix in (".md", ".txt"):
        path = prompts_dir / f"{name}{suffix}"
        try:
            return path.read_text()
        except (FileNotFoundError, OSError):
            continue
    raise FileNotFoundError(
        f"No prompt template {name!r} (.md or .txt) in {prompts_dir}"
    )


def fetch_prompt(name: str, variables: dict[str, object]) -> str:
    template = _load_prompt_template(name)
    for key, value in variables.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def build_themes_context(graph: Graph, empty_msg: str = "(No themes yet.)") -> str:
    """List every theme by name — a menu, not a rendering.

    Previously themes were pre-ranked by an agent-maintained ``priority`` field
    and rendered in full until a token budget filled, demoting the rest to a
    bare name (see git history for `render_hot_themes`). That required global
    re-ranking on every theme-run and a budget number with no principled value.
    Theme names are designed to be self-describing search entry points (see
    weavy-theme.md), so the menu alone is enough to orient — the agent pulls a
    theme's full `state` on demand via `get_theme` when a name looks relevant.
    """
    names = [t.name for t in store_themes.list_all_themes(graph)]
    if not names:
        return empty_msg
    return "\n".join(f"- {name}" for name in names)


def build_themed_system_prompt(
    prompt_name: str,
    graph: Graph,
    system_state: SystemState,
    empty_themes_message: str,
    current_time: str,
    variables: dict[str, object] | None = None,
    caller_context: str | None = None,
) -> str:
    prompt_variables = dict(variables or {})
    prompt_variables["current_time"] = current_time
    prompt_variables["preface"] = (
        system_state.preface or "(not set — call set_preface to describe this graph)"
    )
    prompt_variables["themes_context"] = build_themes_context(
        graph, empty_msg=empty_themes_message
    )
    prompt_variables["caller_context"] = (
        f"- **Caller context:** {caller_context}" if caller_context else ""
    )
    return fetch_prompt(prompt_name, prompt_variables)


def render_full_theme_map(themes: list[Theme]) -> str:
    """Full theme map for the theme agent's own view (every theme, name order)."""
    if not themes:
        return "CURRENT THEME MAP: (empty — no themes yet)\n"

    lines = ["CURRENT THEME MAP:\n"]
    for theme in themes:
        lines.append(f"{theme.render_block()}\n")
    return "\n".join(lines)


def render_session_journal(sessions: list) -> str:
    lines = [f"Sessions since your last run ({len(sessions)}):\n"]
    for i, session in enumerate(sessions, 1):
        changes = session.graph_changes
        change_parts = []
        for action in ("created", "updated", "deleted"):
            node_ids = changes.get(f"nodes_{action}", [])
            edge_ids = changes.get(f"edges_{action}", [])
            if node_ids:
                change_parts.append(f"{action} {', '.join(node_ids)}")
            if edge_ids:
                change_parts.append(f"{action} edges {', '.join(edge_ids)}")
        change_text = "; ".join(change_parts) if change_parts else "no graph changes"
        lines.append(f'{i}. {session.id}: "{session.summary}" — {change_text}')
    return "\n".join(lines)
