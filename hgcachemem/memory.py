"""
HGCacheMem Memory Engine — graph store + vector cache wiring.

Merges vector similarity searches with a knowledge graph to improve
contextual coherence and retrieval speed in agentic systems. Provides
helper factories for Neo4j and Chroma, plus the fallback retrieval
function that pulls context from the graph when the Graph Validator
blocks a cache hit.
"""

from __future__ import annotations

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def connect_graph(
    url: str = "bolt://localhost:7687",
    username: str = "neo4j",
    password: str | None = None,
) -> Neo4jGraph:
    """Return a connected :class:`Neo4jGraph` instance."""
    return Neo4jGraph(url=url, username=username, password=password)


def connect_cache(
    collection_name: str = "semantic_cache",
    persist_directory: str = "./chroma_db",
    embedding_model: str = "all-MiniLM-L6-v2",
    embeddings: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    """Return a :class:`Chroma` vector store backed by HuggingFace embeddings."""
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def get_context_neighbors(
    graph: Neo4jGraph,
    active_entity: str,
    limit: int = 10,
) -> str:
    """
    Fallback retrieval: fetch neighbours of *active_entity* from the graph
    and return them as a formatted context string for an LLM.
    """
    query = """
    MATCH (n {name: $name})-[r]-(neighbor)
    RETURN neighbor.name AS text, type(r) AS relationship
    LIMIT $limit
    """
    try:
        results = graph.query(query, params={"name": active_entity, "limit": limit})
        return "".join(
            f"- Related to {r['relationship']}: {r['text']}\n" for r in results
        )
    except Exception:
        return ""


def get_known_entities(graph: Neo4jGraph) -> list[str]:
    """Return all entity names present in the graph."""
    result = graph.query(
        "MATCH (n) WHERE n.name IS NOT NULL RETURN n.name AS name"
    )
    return [r["name"] for r in result]
