from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.theme import Theme

theme = Theme({"prompt": "bold green", "info": "dim cyan", "err": "bold red"})
console = Console(theme=theme)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Corrective RAG Agent — query a Bedrock Knowledge Base with automatic "
        "relevance evaluation and web-search fallback.",
    )
    parser.add_argument("-q", "--query", help="Run a single query and exit")
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use the simple (non-corrective) RAG agent",
    )
    parser.add_argument(
        "--provider",
        choices=["bedrock", "anthropic", "openai", "ollama"],
        default=None,
        help="LLM provider (overrides AGENT_PROVIDER env var)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID (overrides AGENT_MODEL_ID env var)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    return parser


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def _initialise(
    simple: bool = False,
    provider: str | None = None,
    model_id: str | None = None,
):
    """Load settings, build the model, configure tools, and return the agent."""
    from rag.clients import create_agent_model
    from rag.config import get_settings

    overrides: dict = {}
    if provider:
        overrides["agent_provider"] = provider
    if model_id:
        overrides["agent_model_id"] = model_id

    settings = get_settings(**overrides)
    settings.apply_env()

    agent_model = create_agent_model(settings)

    if not simple:
        if not settings.tavily_api_key:
            console.print(
                "[err]TAVILY_API_KEY is required for the corrective agent. "
                "Set it in .env or as an environment variable.[/err]"
            )
            sys.exit(1)

        from rag.clients import create_evaluation_llm
        from rag.tools import configure_relevance, configure_search

        eval_llm = create_evaluation_llm(settings)
        configure_relevance(eval_llm, settings.relevance_threshold)
        configure_search(max_results=3)

        from rag.agents.corrective import create_agent
    else:
        from rag.agents.simple import create_agent

    return create_agent(model=agent_model)


def _interactive(agent) -> None:
    console.print("\n[bold]Corrective RAG Agent[/bold]")
    console.print("[info]Type 'exit' or 'quit' to stop.[/info]\n")

    while True:
        try:
            query = console.input("[prompt]> [/prompt]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[info]Goodbye.[/info]")
            break

        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue

        result = agent(query)
        console.print(f"\n{result}\n")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(args.verbose)

    agent = _initialise(
        simple=args.simple,
        provider=args.provider,
        model_id=args.model,
    )

    if args.query:
        result = agent(args.query)
        console.print(result)
    else:
        _interactive(agent)


if __name__ == "__main__":
    main()
