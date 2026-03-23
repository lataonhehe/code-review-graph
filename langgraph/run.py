"""Run the LangGraph agent demo (OpenRouter + code-review-graph SQLite graph).

Usage (from repository root):
  uv sync --extra langgraph-agent
  uv run code-review-graph build
  set OPENROUTER_* and LLM_PROVIDER=openrouter
  uv run python langgraph/run.py
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from node import (
    code_agent,
    impact_agent,
    qa_agent,
    retrieve_context,
    review_agent,
    route_after_retrieve,
    route_intent,
)
from state import AgentState


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router", route_intent)
    g.add_node("retrieve", retrieve_context)
    g.add_node("review", review_agent)
    g.add_node("impact", impact_agent)
    g.add_node("qa", qa_agent)
    g.add_node("code", code_agent)

    g.set_entry_point("router")
    g.add_edge("router", "retrieve")

    g.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "review": "review",
            "impact": "impact",
            "qa": "qa",
            "code": "code",
        },
    )

    for name in ("review", "impact", "qa", "code"):
        g.add_edge(name, END)

    return g.compile()


def invoke_agent(
    query: str,
    changed_files: list[str],
    *,
    file_contents: dict[str, str] | None = None,
    repo_root: str | None = None,
    use_git_changed_files: bool = False,
    messages: list | None = None,
) -> dict:
    """Run the compiled graph once and return the final state dict."""
    app = build_graph()
    state: AgentState = {
        "query": query,
        "changed_files": changed_files,
        "file_contents": file_contents or {},
        "messages": messages or [],
    }
    if repo_root is not None:
        state["repo_root"] = repo_root
    if use_git_changed_files:
        state["use_git_changed_files"] = True
    return app.invoke(state)


def main() -> None:
    result = invoke_agent(
        "Review these changes for obvious issues.",
        ["code_review_graph/graph.py"],
    )
    print(result.get("final_answer", result))


if __name__ == "__main__":
    main()
