"""
LangChain Integration Example — HGCacheMem
============================================

Shows how to plug the GraphValidator into a LangChain-based RAG pipeline.

Prerequisites:
    pip install hgcachemem
    # Neo4j running on bolt://localhost:7687
    # OPENAI_API_KEY set in environment (for the fallback LLM)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from hgcachemem import (
    GraphValidator,
    SessionState,
    connect_cache,
    connect_graph,
    get_context_neighbors,
    get_known_entities,
    generate_response,
    resolve_anchor,
)

load_dotenv()

# ── 1. Connect to stores ─────────────────────────────────────────────
graph = connect_graph(
    url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD"),
)
cache = connect_cache(persist_directory="./chroma_db")

# ── 2. Create the GraphValidator ─────────────────────────────────────────
graph_validator = GraphValidator(graph=graph, weight_threshold=0.6)

# ── 3. Set up session state ──────────────────────────────────────────
session = SessionState(user_id="demo_user")
known_entities = get_known_entities(graph)

# ── 4. Shared LLM (optional — avoids creating one per call) ──────────
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


def ask(user_query: str) -> str:
    """
    Full query pipeline:
      1. Resolve anchor (which entity is the user talking about?)
      2. Vector similarity search in the cache
      3. GraphValidator validation against the graph
      4. Return cached answer or fall back to LLM generation
    """
    # — Anchor resolution —
    detected = resolve_anchor(user_query, known_entities, llm=llm)
    if detected:
        session.update_anchor(detected)

    if session.active_entity_id is None:
        return "Which project are you asking about?"

    # — Vector search —
    results = cache.similarity_search(user_query, k=1)

    if results:
        candidate = results[0]
        candidate_entity = candidate.metadata.get("linked_entity")

        if candidate_entity:
            # — GraphValidator check —
            if graph_validator.validate(session.active_entity_id, candidate_entity):
                return candidate.metadata["answer"]  # Cache hit!

    # — Fallback: generate from graph context —
    context = get_context_neighbors(graph, session.active_entity_id)
    return generate_response(user_query, context, llm=llm)


# ── 5. Interactive loop ──────────────────────────────────────────────
if __name__ == "__main__":
    print("HGCacheMem LangChain Demo")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        answer = ask(query)
        print(f"AI:  {answer}\n")
