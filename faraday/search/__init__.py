"""Search tool registry with optional dependency gating.

Package layout under ``faraday.search``:

- ``files/`` — search over in-process tracked files.
- ``web/`` — Exa-backed web and literature tools.

Conversation/memory search helpers live in ``faraday.memory.search_memory`` and are
registered here alongside file and web tools.

Missing optional packages (``exa_py``) must not prevent the core
agent from starting; each cluster is imported in its own
``try`` / ``except ModuleNotFoundError`` block.
"""

ALL_SEARCH_TOOLS = []
ALL_SEARCH_FNS = []
SEARCH_TOOL_DICT = {}

# ---------------------------------------------------------------------------
# Core memory + file search tools (always available)
# ---------------------------------------------------------------------------
# from faraday.search.files.search_filebase import (
#     search_filebase_fn,
#     search_filebase_tool,
# )
# # from faraday.memory.search_memory import (
# #     get_project_summary_fn,
# #     get_project_summary_tool,
# #     search_chat_fn,
# #     search_chat_tool,
# #     search_memory_fn,
# #     search_memory_tool,
# # )

# # ALL_SEARCH_TOOLS.extend(
# #     [
# #         search_filebase_tool,
# #         search_memory_tool,
# #         search_chat_tool,
# #         get_project_summary_tool,
# #     ]
# # )
# # ALL_SEARCH_FNS.extend(
# #     [
# #         search_filebase_fn,
# #         search_memory_fn,
# #         search_chat_fn,
# #         get_project_summary_fn,
# #     ]
# )
# # SEARCH_TOOL_DICT.update(
# #     {
# #         "search_filebase": search_filebase_fn,
# #         "search_memory": search_memory_fn,
# #         "get_project_summary": get_project_summary_fn,
# #         "search_chat": search_chat_fn,
# #     }
# # )

# ---------------------------------------------------------------------------
# Exa: faraday/search/web/
# ---------------------------------------------------------------------------
try:
    # literature → pharma → fetch → structures → open web / QA
    from faraday.search.web.scientific_literature_search import (
        scientific_literature_search,
        scientific_literature_search_tool,
    )
    from faraday.search.web.search_pharma_web import (
        pharma_web_search,
        pharma_web_search_tool,
    )
    from faraday.search.web.read_webpage_exa import (
        read_webpage,
        read_webpage_tool,
    )
    from faraday.search.web.get_SMILES import (
        name_to_smiles,
        name_to_smiles_tool,
    )
    from faraday.search.web.search_web import (
        general_web_search,
        general_web_search_tool,
    )
    from faraday.search.web.search_web_qa import (
        general_web_search_question_answering,
        general_web_search_question_answering_tool,
    )
except ModuleNotFoundError as exc:
    if exc.name != "exa_py":
        raise
else:
    ALL_SEARCH_TOOLS.extend(
        [
            scientific_literature_search_tool,
            pharma_web_search_tool,
            read_webpage_tool,
            name_to_smiles_tool,
            general_web_search_tool,
            general_web_search_question_answering_tool,
        ]
    )
    ALL_SEARCH_FNS.extend(
        [
            scientific_literature_search,
            pharma_web_search,
            read_webpage,
            name_to_smiles,
            general_web_search,
            general_web_search_question_answering,
        ]
    )
    SEARCH_TOOL_DICT.update(
        {
            "scientific_literature_search": scientific_literature_search,
            "pharma_web_search": pharma_web_search,
            "read_webpage": read_webpage,
            "name_to_smiles": name_to_smiles,
            "general_web_search": general_web_search,
            "general_web_search_question_answering": general_web_search_question_answering,
        }
    )
