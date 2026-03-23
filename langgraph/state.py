"""Agent state for the LangGraph demo (OpenRouter + code-review-graph)."""

from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, Required, TypedDict


class AgentState(TypedDict, total=False):
    """State flowing through router → retrieve → specialist agents."""

    query: Required[str]
    changed_files: Required[list[str]]
    file_contents: Required[dict[str, str]]
    messages: Required[Annotated[list, add_messages]]
    repo_root: NotRequired[str | None]
    use_git_changed_files: NotRequired[bool]
    intent: NotRequired[str]
    blast_radius: NotRequired[dict]
    relevant_context: NotRequired[str]
    final_answer: NotRequired[str]
