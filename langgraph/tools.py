"""LangChain tools backed by code_review_graph (no fictional CodeGraph)."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from code_review_graph.tools import get_impact_radius, query_graph, semantic_search_nodes


def _blast_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Map get_impact_radius output to a stable shape for retrieve_context."""
    if raw.get("status") != "ok":
        return {
            "status": raw.get("status", "error"),
            "error": raw.get("summary") or raw.get("error", "unknown"),
            "affected_files": [],
            "affected_nodes": [],
            "test_coverage": 0,
            "summary_tokens": 0,
            "graph_summary": "",
        }
    impacted_nodes = raw.get("impacted_nodes") or []
    test_hits = sum(
        1
        for n in impacted_nodes
        if isinstance(n, dict)
        and (n.get("kind") == "Test" or n.get("is_test") is True)
    )
    summary = raw.get("summary") or ""
    return {
        "status": "ok",
        "graph_summary": summary,
        "affected_files": list(raw.get("impacted_files") or []),
        "affected_nodes": impacted_nodes[:500],
        "test_coverage": test_hits,
        "summary_tokens": max(1, len(summary) // 4),
        "truncated": bool(raw.get("truncated", False)),
    }


@tool
def get_blast_radius(
    changed_files: list[str] | None = None,
    repo_root: str | None = None,
    max_depth: int = 2,
) -> str:
    """Blast radius: files/nodes affected by changes. Pass None to auto-detect from git."""
    raw = get_impact_radius(
        changed_files=changed_files,
        max_depth=max_depth,
        repo_root=repo_root,
    )
    return json.dumps(_blast_payload(raw))


@tool
def get_callers(function_qualified_name: str, repo_root: str | None = None) -> str:
    """Find call sites for a function (qualified name, e.g. path.py::Class.method)."""
    raw = query_graph("callers_of", function_qualified_name, repo_root=repo_root)
    return json.dumps(raw)


@tool
def get_dependencies(file_path: str, repo_root: str | None = None) -> str:
    """Imports from a file (outgoing import edges)."""
    raw = query_graph("imports_of", file_path, repo_root=repo_root)
    return json.dumps(raw)


@tool
def search_code(query: str, repo_root: str | None = None) -> str:
    """Keyword or semantic search over graph nodes (embeddings optional)."""
    raw = semantic_search_nodes(query, limit=5, repo_root=repo_root)
    return json.dumps(raw)
