from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from strands import tool

logger = logging.getLogger(__name__)

_search_tool: TavilySearchResults | None = None


def configure(max_results: int = 3) -> None:
    """Initialise the Tavily search client. Reads TAVILY_API_KEY from env."""
    global _search_tool
    _search_tool = TavilySearchResults(k=max_results)


@tool
def web_search(query: str) -> dict:
    """
    Perform web search based on the query and return results as Documents.
    Only to be used if chunk_relevance_score is no.

    Args:
        query: The user question or rephrased query.

    Returns:
        dict: documents list containing web search results.
    """
    if _search_tool is None:
        raise RuntimeError("Search tool not configured. Call configure() first.")

    logger.info("Performing web search for: %s", query)

    raw = _search_tool.invoke({"query": query})
    documents = [Document(page_content=d["content"]) for d in raw]

    return {"documents": documents}
