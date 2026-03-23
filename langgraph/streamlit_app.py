"""Streamlit UI for the LangGraph + OpenRouter + code-review-graph demo.

Run from repository root:

    uv sync --extra langgraph-agent
    uv run streamlit run langgraph/streamlit_app.py

Requires ``OPENROUTER_API_KEY`` (and optional ``OPENROUTER_*``) in the environment
or in a ``.env`` file at the repo root. Build the graph first:
``uv run code-review-graph build``.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from run import build_graph

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@st.cache_resource
def get_compiled_graph():
    return build_graph()


def _parse_paths(text: str) -> list[str]:
    lines = []
    for line in text.replace(",", "\n").splitlines():
        s = line.strip()
        if s:
            lines.append(s)
    return lines


def main() -> None:
    st.set_page_config(
        page_title="Code Review Graph — LangGraph",
        page_icon="◇",
        layout="wide",
    )
    repo_default = Path(__file__).resolve().parent.parent

    st.title("Code review graph agent")
    st.caption("LangGraph router → blast radius (SQLite) → OpenRouter LLM")

    with st.sidebar:
        st.header("Settings")
        repo_root = st.text_input(
            "Repository root",
            value=str(repo_default),
            help="Project with `.code-review-graph/graph.db` (run `code-review-graph build`).",
        ).strip()
        use_git = st.checkbox(
            "Detect changed files from git",
            value=False,
            help="Uses git diff when enabled; ignores the path list below for blast radius.",
        )
        key_set = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
        st.caption(
            "OpenRouter API key: " + ("set" if key_set else "missing — set OPENROUTER_API_KEY")
        )

    col_q, col_f = st.columns((1, 1))
    with col_q:
        query = st.text_area(
            "Query",
            value="Review these changes for obvious issues.",
            height=120,
        )
    with col_f:
        paths_raw = st.text_area(
            "Changed files (one per line, relative to repo root)",
            value="code_review_graph/graph.py",
            height=120,
            help="Ignored when “Detect changed files from git” is on.",
        )

    run_btn = st.button("Run graph", type="primary")

    if not run_btn:
        return

    if not key_set:
        st.error(
            "Set `OPENROUTER_API_KEY` in `.env` at the repo root or in the environment, then rerun."
        )
        return

    changed_files = _parse_paths(paths_raw)
    if not use_git and not changed_files:
        st.error("Add at least one changed file path, or enable git detection.")
        return

    root_path = Path(repo_root)
    if not root_path.is_dir():
        st.error(f"Repository root is not a directory: {repo_root}")
        return

    state: dict = {
        "query": query.strip(),
        "changed_files": changed_files,
        "file_contents": {},
        "messages": [],
        "repo_root": str(root_path.resolve()),
    }
    if use_git:
        state["use_git_changed_files"] = True

    graph = get_compiled_graph()
    steps: list[tuple[str, dict]] = []

    try:
        with st.status("Running LangGraph…", expanded=True) as status:
            last: dict = {}
            for step in graph.stream(state, stream_mode="values"):
                last = step
                intent = step.get("intent")
                node_hint = []
                if intent:
                    node_hint.append(f"intent={intent}")
                if step.get("relevant_context"):
                    node_hint.append("context loaded")
                if step.get("final_answer"):
                    node_hint.append("answer ready")
                label = ", ".join(node_hint) if node_hint else "state update"
                status.write(label)
                steps.append((label, step))
            status.update(label="Done", state="complete")

        ans = last.get("final_answer")
        if ans:
            st.subheader("Answer")
            st.markdown(str(ans))
        else:
            st.warning("No final answer in graph output.")
            st.json({k: v for k, v in last.items() if k != "messages"})

        with st.expander("Retrieval context (blast radius + file snippets)", expanded=False):
            st.text(last.get("relevant_context") or "(empty)")

        with st.expander("Blast radius (structured)", expanded=False):
            st.json(last.get("blast_radius") or {})

        with st.expander("Execution steps", expanded=False):
            for i, (label, snap) in enumerate(steps):
                st.markdown(f"**Step {i + 1}:** {label}")
                if snap.get("intent"):
                    st.caption(f"Intent: `{snap['intent']}`")

    except OSError as e:
        st.error(str(e))
    except Exception:
        st.exception("Graph run failed")


if __name__ == "__main__":
    main()
