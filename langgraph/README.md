# LangGraph demo (OpenRouter + code-review-graph)

Optional demo that runs a small **LangGraph** workflow on top of the same SQLite knowledge graph as the MCP server. It classifies the user query, pulls **blast-radius** context via `code_review_graph.tools`, then calls an **OpenRouter**-compatible chat model (`langchain_openai.ChatOpenAI`).

This directory is **not** installed as a Python package (no `__init__.py` here) so the name does not shadow the PyPI `langgraph` library. Run scripts from the **repository root** as shown below.

## Flow

```text
router (intent: review | impact | qa | code)
  → retrieve (get_impact_radius → read up to 15 affected files)
  → specialist node (review / impact / qa / code)
  → END
```

## Prerequisites

1. Python 3.10+ and [uv](https://docs.astral.sh/uv/).
2. Install extras and the project in editable mode:

   ```bash
   uv sync --extra langgraph-agent
   ```

3. Build the graph for the repo you want to analyze:

   ```bash
   uv run code-review-graph build
   ```

4. OpenRouter credentials (see [Environment variables](#environment-variables)).

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | No | Default `openrouter`. Only `openrouter` is implemented. |
| `OPENROUTER_API_KEY` | Yes | API key from [OpenRouter](https://openrouter.ai/). |
| `OPENROUTER_MODEL` | No | Default `nvidia/nemotron-3-nano-30b-a3b:free`. |
| `OPENROUTER_BASE_URL` | No | Default `https://openrouter.ai/api/v1`. |
| `OPENROUTER_HTTP_REFERER` | No | Optional; used in `default_headers` for OpenRouter. |
| `OPENROUTER_APP_TITLE` | No | Optional app title header for OpenRouter. |

`streamlit_app.py` loads a `.env` file from the **repository root** if present.

## CLI

From the repository root:

```bash
# Windows PowerShell example
$env:LLM_PROVIDER = "openrouter"
$env:OPENROUTER_API_KEY = "your-key"
$env:OPENROUTER_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
$env:OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

uv run python langgraph/run.py
```

The default `main()` uses a sample query and `changed_files: ["code_review_graph/graph.py"]`. Edit `run.py` or call `invoke_agent()` from your own script.

### Programmatic use

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("langgraph").resolve()))

from run import invoke_agent

out = invoke_agent(
    "Review these changes for obvious issues.",
    ["code_review_graph/graph.py"],
    repo_root=str(Path.cwd()),
)
print(out.get("final_answer"))
```

Use `use_git_changed_files=True` to let `get_impact_radius` infer changed files from git (same behaviour as the core tool’s `changed_files=None` path).

## Streamlit UI

From the repository root:

```bash
uv run streamlit run langgraph/streamlit_app.py
```

The app lets you set **repository root**, **query**, **changed files** (one path per line, relative to repo root), and optionally **detect changed files from git**. It streams graph state updates (`stream_mode="values"`) and shows the final answer plus expandable retrieval context.

## Files

| File | Role |
|------|------|
| `run.py` | Builds and compiles the graph; `invoke_agent()`; CLI entry. |
| `streamlit_app.py` | Streamlit front-end. |
| `state.py` | `AgentState` TypedDict (including optional `repo_root`, `use_git_changed_files`). |
| `node.py` | Nodes: intent router, `retrieve_context`, four LLM agents; OpenRouter `ChatOpenAI`. |
| `tools.py` | LangChain `@tool` wrappers around `get_impact_radius`, `query_graph`, `semantic_search_nodes`. |

## Dependencies

Declared under the **`langgraph-agent`** optional extra in the root `pyproject.toml` (`langgraph`, `langchain-core`, `langchain-openai`, `streamlit`, `python-dotenv`, `typing_extensions`, …).

## Troubleshooting

- **`OPENROUTER_API_KEY is not set`**: Export the variable or add it to repo-root `.env` before running the CLI or Streamlit app.
- **Import errors when running `python langgraph/run.py`**: Run from the **repository root** so the script directory (`langgraph/`) is on `sys.path` and sibling imports (`from run import …`) resolve. Streamlit adds the script’s directory automatically.
- **Empty or wrong blast radius**: Run `uv run code-review-graph build` in the target repo; ensure `repo_root` points at a directory that contains `.code-review-graph/graph.db` (or `.git` for discovery). Paths in `changed_files` must be **relative to that repo root** unless you use git detection.
