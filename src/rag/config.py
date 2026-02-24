from __future__ import annotations

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    knowledge_base_id: str = Field(description="Amazon Bedrock Knowledge Base ID")
    aws_region: str = Field(default="us-east-1")
    tavily_api_key: str | None = Field(default=None)
    min_score: float = Field(default=0.2)

    # Agent LLM
    agent_provider: str = Field(default="bedrock")
    agent_model_id: str | None = Field(default=None)

    # Anthropic direct API
    anthropic_api_key: str | None = Field(default=None)

    # OpenAI-compatible API
    openai_api_key: str | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)

    # Ollama
    ollama_host: str | None = Field(default="http://localhost:11434")

    # Evaluation / embeddings
    eval_model_id: str = Field(default="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    embedding_model_id: str = Field(default="amazon.titan-embed-text-v2:0")
    relevance_threshold: float = Field(default=0.5)

    def apply_env(self) -> None:
        """Propagate settings to environment variables consumed by strands_tools."""
        os.environ["KNOWLEDGE_BASE_ID"] = self.knowledge_base_id
        os.environ["AWS_REGION"] = self.aws_region
        os.environ["MIN_SCORE"] = str(self.min_score)
        if self.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = self.tavily_api_key


def get_settings(**overrides: object) -> Settings:
    """Build a Settings instance, merging .env file with any explicit overrides."""
    return Settings(**overrides)  # type: ignore[arg-type]
