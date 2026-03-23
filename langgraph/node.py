"""LangGraph nodes: intent router, graph-backed retrieval, specialist LLM agents."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from state import AgentState
from tools import get_blast_radius

load_dotenv()

_ALLOWED_INTENTS = frozenset({"review", "impact", "qa", "code"})


def _build_llm() -> ChatOpenAI:
    provider = os.environ.get("LLM_PROVIDER", "openrouter").strip().lower()
    if provider != "openrouter":
        raise ValueError(
            f"Unsupported LLM_PROVIDER={provider!r}; only 'openrouter' is implemented."
        )
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise OSError(
            "OPENROUTER_API_KEY is not set. Export it before running langgraph/run.py."
        )
    model = os.environ.get(
        "OPENROUTER_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free"
    )
    base_url = os.environ.get(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    referer = os.environ.get(
        "OPENROUTER_HTTP_REFERER",
        "https://github.com/tirth8205/code-review-graph",
    )
    title = os.environ.get("OPENROUTER_APP_TITLE", "code-review-graph-langgraph")
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
    }
    # OpenRouter optional ranking headers (supported by underlying OpenAI client).
    try:
        return ChatOpenAI(
            **kwargs,
            default_headers={"HTTP-Referer": referer, "X-Title": title},
        )
    except TypeError:
        return ChatOpenAI(**kwargs)


_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = _build_llm()
    return _llm


def _normalize_intent(text: str) -> str:
    raw = (text or "").strip().lower()
    token = re.split(r"\W+", raw)[0] if raw else ""
    if token in _ALLOWED_INTENTS:
        return token
    for word in _ALLOWED_INTENTS:
        if word in raw:
            return word
    return "qa"


def route_intent(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify user intent: review | impact | qa | code. Reply with one word only.",
            ),
            ("human", "{query}"),
        ]
    )
    result = (prompt | get_llm()).invoke({"query": state["query"]})
    content = getattr(result, "content", "") or ""
    return {**state, "intent": _normalize_intent(str(content))}


def retrieve_context(state: AgentState) -> AgentState:
    if state.get("use_git_changed_files"):
        cf_arg: list[str] | None = None
    else:
        cf_arg = state["changed_files"]
    raw = get_blast_radius.invoke(
        {
            "changed_files": cf_arg,
            "repo_root": state.get("repo_root"),
            "max_depth": 2,
        }
    )
    blast = json.loads(str(raw))

    affected = blast.get("affected_files") or []
    relevant_files = affected[:15]
    contents: dict[str, str] = {}
    for fpath in relevant_files:
        p = Path(fpath)
        try:
            contents[fpath] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

    graph_summary = blast.get("graph_summary") or ""
    context = f"""
Blast radius summary (~{blast.get("summary_tokens", 0)} token est.):
{graph_summary}

Affected nodes (truncated list): {len(blast.get("affected_nodes") or [])}
Test-related nodes in radius: {blast.get("test_coverage", 0)}

Relevant source files ({len(contents)}/{len(relevant_files)} read):
"""
    for path, src in contents.items():
        context += f"\n--- {path} ---\n{src[:2000]}\n"

    return {**state, "blast_radius": blast, "relevant_context": context}


REVIEW_PROMPT = """You are a senior code reviewer.
Context (structural graph + source):
{context}

Changed files: {changed_files}

Review for: bugs, security issues, performance, style.
Be specific — cite file:line.
Trả lời bằng tiếng Việt."""

IMPACT_PROMPT = """Analyze the blast radius of these changes.
Graph data: {context}
Identify: breaking changes, tests that must be updated, downstream risk.
Trả lời bằng tiếng Việt."""

QA_PROMPT = """Answer this question about the codebase.
Context: {context}
Question: {query}
Trả lời bằng tiếng Việt."""

CODE_PROMPT = """You are a coding agent with full codebase context.
Context: {context}
Task: {query}
Produce the minimal, correct change."""


def make_agent(prompt_template: str):
    def agent_node(state: AgentState) -> AgentState:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template),
                ("human", "Go."),
            ]
        )
        chain = prompt | get_llm()
        result = chain.invoke(
            {
                "context": state.get("relevant_context", ""),
                "changed_files": state.get("changed_files", []),
                "query": state.get("query", ""),
            }
        )
        return {**state, "final_answer": result.content, "messages": [result]}

    return agent_node


review_agent = make_agent(REVIEW_PROMPT)
impact_agent = make_agent(IMPACT_PROMPT)
qa_agent = make_agent(QA_PROMPT)
code_agent = make_agent(CODE_PROMPT)


def route_after_retrieve(state: AgentState) -> str:
    intent = state.get("intent") or "qa"
    return intent if intent in _ALLOWED_INTENTS else "qa"
