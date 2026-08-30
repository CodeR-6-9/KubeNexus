import logfire
from langchain_groq import ChatGroq

from app.config import settings
from app.guardrails.colang_rules import (
    build_classification_prompt,
    CATEGORY_RESPONSES,
    VALID_CATEGORIES,
)

_guard_llm: ChatGroq | None = None


def initialize_rails() -> None:
    """Build the guard-gate LLM singleton at app startup."""
    global _guard_llm
    _guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="openai/gpt-oss-20b",
        temperature=0,
    )
    logfire.info("Guardrail classifier initialised (model=openai/gpt-oss-20b).")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the structured classification gate.

    Returns:
        (True,  response) — message falls in a handled category
                             (off_topic, jailbreak, greeting, capabilities,
                             farewell); return this response, skip RAG.
        (False, None)     — message is "clean"; proceed to LangGraph.
    """
    if _guard_llm is None:
        logfire.warning("Guardrail classifier not initialised — skipping gate.")
        return False, None

    with logfire.span("Guardrails Check"):
        result = _guard_llm.invoke(build_classification_prompt(message))
        category = result.content.strip().lower()

        if category not in VALID_CATEGORIES:
            logfire.warning(f"Unrecognized category '{category}' from guard LLM — defaulting to clean.")
            category = "clean"

        if category == "clean":
            logfire.info("Guardrails passed.")
            return False, None

        logfire.info(f"Guardrails fired | category={category} | query='{message[:80]}'")
        return True, CATEGORY_RESPONSES[category]