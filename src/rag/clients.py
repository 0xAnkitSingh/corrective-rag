from __future__ import annotations

import logging

import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from strands.models import Model

from rag.config import Settings

logger = logging.getLogger(__name__)

_PROVIDER_DEFAULTS: dict[str, str] = {
    "bedrock": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "ollama": "llama3",
}


def create_agent_model(settings: Settings) -> Model | str:
    """Build a Strands-compatible model from the configured provider.

    Supports: bedrock (default), anthropic, openai (+ any compatible endpoint), ollama.
    """
    provider = settings.agent_provider.lower()
    model_id = settings.agent_model_id or _PROVIDER_DEFAULTS.get(provider)

    if provider == "bedrock":
        from strands.models.bedrock import BedrockModel

        kwargs: dict = {"region_name": settings.aws_region}
        if model_id:
            kwargs["model_id"] = model_id
        logger.info("Using Bedrock model: %s in %s", model_id, settings.aws_region)
        return BedrockModel(**kwargs)

    if provider == "anthropic":
        from strands.models.anthropic import AnthropicModel

        client_args: dict = {}
        if settings.anthropic_api_key:
            client_args["api_key"] = settings.anthropic_api_key
        logger.info("Using Anthropic model: %s", model_id)
        return AnthropicModel(client_args=client_args, model_id=model_id)

    if provider == "openai":
        from strands.models.openai import OpenAIModel

        client_args = {}
        if settings.openai_api_key:
            client_args["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            client_args["base_url"] = settings.openai_base_url
        logger.info("Using OpenAI-compatible model: %s", model_id)
        return OpenAIModel(client_args=client_args, model_id=model_id)

    if provider == "ollama":
        from strands.models.ollama import OllamaModel

        host = settings.ollama_host or "http://localhost:11434"
        logger.info("Using Ollama model: %s at %s", model_id, host)
        return OllamaModel(host=host, model_id=model_id)

    raise ValueError(
        f"Unknown agent provider '{provider}'. "
        f"Supported: {', '.join(_PROVIDER_DEFAULTS)}"
    )


# ── Evaluation / embeddings (always Bedrock) ──────────────────────────────────


def create_evaluation_llm(settings: Settings) -> LangchainLLMWrapper:
    llm = ChatBedrockConverse(
        model_id=settings.eval_model_id,
        region_name=settings.aws_region,
        additional_model_request_fields={"thinking": {"type": "disabled"}},
    )
    return LangchainLLMWrapper(llm)


def create_embeddings(settings: Settings) -> LangchainEmbeddingsWrapper:
    embeddings = BedrockEmbeddings(
        model_id=settings.embedding_model_id,
        region_name=settings.aws_region,
    )
    return LangchainEmbeddingsWrapper(embeddings)


def create_bedrock_agent_runtime(settings: Settings) -> boto3.client:
    return boto3.client("bedrock-agent-runtime", region_name=settings.aws_region)
