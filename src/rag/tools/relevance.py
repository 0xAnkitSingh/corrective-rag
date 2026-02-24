from __future__ import annotations

import asyncio
import logging
import re

from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference
from strands import tool

logger = logging.getLogger(__name__)

_evaluation_llm: LangchainLLMWrapper | None = None
_relevance_threshold: float = 0.5


def configure(evaluation_llm: LangchainLLMWrapper, relevance_threshold: float = 0.5) -> None:
    global _evaluation_llm, _relevance_threshold
    _evaluation_llm = evaluation_llm
    _relevance_threshold = relevance_threshold


@tool
def check_chunks_relevance(results: str, question: str) -> dict:
    """
    Evaluates the relevance of retrieved chunks to the user question using RAGAS.

    Args:
        results: Retrieval output as a string with 'Score:' and 'Content:' patterns.
        question: Original user question.

    Returns:
        dict: A binary score ('yes' or 'no') and the numeric relevance score, or an error message.
    """
    if _evaluation_llm is None:
        raise RuntimeError("Relevance tool not configured. Call configure() first.")

    try:
        if not results or not isinstance(results, str):
            raise ValueError("'results' must be a non-empty string.")
        if not question or not isinstance(question, str):
            raise ValueError("'question' must be a non-empty string.")

        pattern = r"Score:.*?\nContent:\s*(.*?)(?=Score:|\Z)"
        docs = [chunk.strip() for chunk in re.findall(pattern, results, re.DOTALL)]

        if not docs:
            raise ValueError("No valid content chunks found in 'results'.")

        sample = SingleTurnSample(
            user_input=question,
            response="placeholder-response",
            retrieved_contexts=docs,
        )

        scorer = LLMContextPrecisionWithoutReference(llm=_evaluation_llm)
        score = asyncio.run(scorer.single_turn_ascore(sample))

        logger.info("Context precision score: %.4f (threshold: %.2f)", score, _relevance_threshold)

        return {
            "chunk_relevance_score": "yes" if score > _relevance_threshold else "no",
            "chunk_relevance_value": score,
        }

    except Exception as exc:
        logger.exception("Chunk relevance evaluation failed")
        return {
            "error": str(exc),
            "chunk_relevance_score": "unknown",
            "chunk_relevance_value": None,
        }
