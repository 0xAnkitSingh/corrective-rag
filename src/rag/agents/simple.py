from __future__ import annotations

from strands import Agent
from strands.models import Model
from strands_tools import retrieve

from rag.agents.prompts import SIMPLE_RAG_PROMPT


def create_agent(model: Model | str | None = None) -> Agent:
    return Agent(
        model=model,
        tools=[retrieve],
        system_prompt=SIMPLE_RAG_PROMPT,
    )
