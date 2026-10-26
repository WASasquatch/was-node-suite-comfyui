"""Language models, documents, indexes, retrieval, chat, memory and agents, on llama-index.

Every node in the family runs behind ``features.llm``. llama-index is imported on first use.
"""

from __future__ import annotations

#: The config key that turns the LLM nodes on.
FEATURE = "features.llm"
