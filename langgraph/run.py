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


def main() -> None:
    app = build_graph()
    result = app.invoke(
        {
            "query": "Review these changes for obvious issues.",
            "changed_files": ["code_review_graph/graph.py"],
            "file_contents": {},
            "messages": [],
        }
    )
    print(result.get("final_answer", result))


if __name__ == "__main__":
    main()
