"""
HGCacheMem — A robust architectural approach to improve contextual
coherence and retrieval speed in agentic systems.

Merges vector similarity searches with a topological Graph Validator that
checks potential cache hits against the user's active context within a
knowledge graph prior to delivery.
"""

from hgcachemem.graph_validator import GraphValidator
from hgcachemem.memory import (
    connect_cache,
    connect_graph,
    get_context_neighbors,
    get_known_entities,
)
from hgcachemem.state import SessionState
from hgcachemem.generation import generate_response, resolve_anchor

__all__ = [
    "GraphValidator",
    "SessionState",
    "connect_cache",
    "connect_graph",
    "get_context_neighbors",
    "get_known_entities",
    "generate_response",
    "resolve_anchor",
]

__version__ = "0.1.1"
