"""
HGCacheMem Generation Engine — anchor resolution and LLM fallback.

Provides the dynamic anchor resolver (extracts the entity from a user
query and matches it against known graph nodes) and the fallback response
generator that is invoked when the Graph Validator blocks a cache hit,
ensuring contextual coherence in agentic systems.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class EntityExtraction(BaseModel):
    """Structured output schema for entity extraction."""
    entity_name: str | None = Field(
        default=None,
        description=(
            "The name of the project or entity the user is referring to. "
            "Return null if no specific entity is mentioned."
        ),
    )


def resolve_anchor(
    user_query: str,
    known_entities: list[str],
    llm: ChatOpenAI | None = None,
) -> str | None:
    """
    Extract the entity the user is talking about and match it to a
    known graph node.

    Uses the LLM's structured output to pull an entity name, then
    attempts an exact (case-insensitive) match followed by a partial
    substring match against *known_entities*.

    Parameters
    ----------
    user_query : str
        The raw user input.
    known_entities : list[str]
        Entity names fetched from the knowledge graph.
    llm : ChatOpenAI, optional
        The LLM to use for extraction.  A default ``gpt-3.5-turbo``
        instance is created if not supplied.

    Returns
    -------
    str or None
        The matched entity name, or ``None`` if nothing matched.
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # --- Phase 1: LLM-based extraction ---
    try:
        extractor = llm.with_structured_output(EntityExtraction)
        result = extractor.invoke(
            f"Extract the project or entity name from this query: {user_query}"
        )
        extracted = result.entity_name if result.entity_name else None
    except Exception:
        extracted = None

    if extracted:
        # Exact match (case-insensitive)
        for entity in known_entities:
            if extracted.lower() == entity.lower():
                return entity
        # Partial / substring match
        for entity in known_entities:
            if extracted.lower() in entity.lower() or entity.lower() in extracted.lower():
                return entity

    # --- Phase 2: Keyword fallback ---
    # If the LLM extraction failed or returned no match, check whether
    # any known entity name (or a distinctive keyword from it) appears
    # directly in the user query.  This makes anchor resolution robust
    # against inconsistent structured-output results.
    query_lower = user_query.lower()
    for entity in known_entities:
        if entity.lower() in query_lower:
            return entity
    # Check individual distinctive words (e.g. "free" → "CloudSync Free")
    for entity in known_entities:
        words = entity.lower().split()
        for word in words:
            if len(word) >= 3 and word in query_lower:
                return entity

    return None


def generate_response(
    user_query: str,
    context: str,
    llm: ChatOpenAI | None = None,
) -> str:
    """
    Fallback generation: produce an LLM answer grounded in graph context.

    Called when the Graph Validator blocks the cached answer and we fall back
    to retrieving raw neighbours from the graph.

    Parameters
    ----------
    user_query : str
        The user's question.
    context : str
        Formatted context string from :func:`memory.get_context_neighbors`.
    llm : ChatOpenAI, optional
        The LLM to use.  Defaults to ``gpt-3.5-turbo``.

    Returns
    -------
    str
        The generated response text.
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    system_prompt = (
        "You are a helpful assistant for a project management system.\n"
        "Use the following CONTEXT information to answer the user's question.\n\n"
        f"CONTEXT from the Graph:\n{context}\n\n"
        "RULES:\n"
        "- If the context contains the answer, use it.\n"
        "- If the context does NOT contain the answer, admit you don't know.\n"
        "- Do NOT make up facts that are not in the context."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    try:
        return llm.invoke(messages).content
    except Exception as e:
        return f"Generation Error: {e}"
