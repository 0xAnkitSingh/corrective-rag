from __future__ import annotations

from strands import Agent
from strands.models import Model
from strands_tools import retrieve

from rag.agents.prompts import CORRECTIVE_RAG_PROMPT
from rag.tools.relevance import check_chunks_relevance
from rag.tools.search import web_search


def create_agent(model: Model | str | None = None) -> Agent:
    return Agent(
        model=model,
        tools=[retrieve, check_chunks_relevance, web_search],
        system_prompt=CORRECTIVE_RAG_PROMPT,
    )
