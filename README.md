# HGCacheMem

**A robust architectural approach to improve contextual coherence and retrieval speed in agentic systems.**

LLM agents often encounter a significant trade-off between retrieval speed and contextual coherence. Semantic caching systems offer sub-millisecond retrieval speeds but are plagued by "context ignorance," resulting in inaccurate cache hits. Graph-based and other complex memory systems yield high contextual accuracy but come with latency drawbacks that impact real-time user experience.

HGCacheMem is a robust architecture that merges vector similarity searches with a topological **Graph Validator**. The Graph Validator checks potential cache hits against the user's active context within a knowledge graph prior to delivery.

Experimental tests on the RGB Noise Robustness benchmark testbed show that HGCacheMem achieves a **contextual coherence accuracy of 97.7%**, significantly surpassing other caching alternatives and complex memory systems, with a **mean retrieval latency of 37.39 ms** — proving that structural validation can be achieved within real-time interaction thresholds.

## How It Works

```
User Query
    │
    ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│ Vector Cache  │────▶│ Graph Validator   │────▶│   Response   │
│ (Chroma)      │     │ (Neo4j path check │     │              │
│ similarity    │     │  weight > 0.6)    │     │ Cache hit OR │
│ search        │     │                   │     │ LLM fallback │
└──────────────┘     └───────────────────┘     └──────────────┘
```

1. A user query hits the **vector cache** (Chroma) via similarity search.
2. The **Graph Validator** runs a Cypher shortest-path query on a Neo4j knowledge graph to verify that the candidate answer is **structurally connected** to the user's current context anchor, with every edge weight exceeding a configurable threshold (default `0.6`).
3. If the graph validation **passes** → return the cached answer (fast path).
4. If it **fails** → fall back to LLM generation grounded in graph context (safe path).

## Benchmark Results

Evaluated on the RGB Noise Robustness benchmark (300 cases):

| Metric | Value |
|---|---|
| Contextual Coherence Accuracy | 97.7% |
| Trapped (cross-context hallucination) | 0.0% |
| Mean Retrieval Latency | 37.39 ms |
| p50 Latency | 26.16 ms |
| p90 Latency | 65.42 ms |
| p95 Latency | 91.64 ms |
| p99 Latency | 142.88 ms|

## Installation

```bash
pip install hgcachemem
```

## Quick Start

```python
from hgcachemem import GraphValidator, connect_graph, connect_cache

# Connect to your stores
graph = connect_graph(url="bolt://localhost:7687", username="neo4j", password="secret")
cache = connect_cache(persist_directory="./chroma_db")

# Create the Graph Validator
graph_validator = GraphValidator(graph=graph, weight_threshold=0.6)

# Validate a cache hit before returning it
results = cache.similarity_search("What is the budget?", k=1)
candidate_entity = results[0].metadata["linked_entity"]

if graph_validator.validate(active_entity="Project Alpha", candidate_entity=candidate_entity):
    print("Cache hit is valid for this context!")
    print(results[0].metadata["answer"])
else:
    print("Blocked — answer belongs to a different context.")
```

## Full Pipeline Example

See [`examples/langchain_example.py`](examples/langchain_example.py) for a complete LangChain integration with anchor resolution, vector search, Graph Validator validation, and LLM fallback.

## API Reference

### `GraphValidator(graph, weight_threshold=0.6, max_hops=2)`

The core validation class.

- **`graph`** — A LangChain `Neo4jGraph` instance.
- **`weight_threshold`** — Minimum edge weight required on every relationship in the path (default `0.6`).
- **`max_hops`** — Maximum path length for the shortest path check (default `2`).

#### `graph_validator.validate(active_entity, candidate_entity) -> bool`

Returns `True` if the candidate is contextually valid for the current anchor.

### `connect_graph(url, username, password) -> Neo4jGraph`

Factory for a connected Neo4j graph instance.

### `connect_cache(collection_name, persist_directory, embedding_model) -> Chroma`

Factory for a Chroma vector store with HuggingFace embeddings.

### `SessionState(user_id)`

Tracks the user's current context anchor across a conversation.

- **`update_anchor(entity_id)`** — Move focus to a new entity.
- **`active_entity_id`** — The current anchor (or `None`).

### `resolve_anchor(user_query, known_entities, llm=None) -> str | None`

Uses an LLM to extract the entity from a query and match it against known graph nodes.

### `generate_response(user_query, context, llm=None) -> str`

Fallback LLM generation grounded in graph-retrieved context.

### `get_context_neighbors(graph, active_entity, limit=10) -> str`

Retrieves neighbours of the active entity from the graph as formatted context.

### `get_known_entities(graph) -> list[str]`

Returns all entity names present in the graph.

## Requirements

- Python >= 3.10
- Neo4j >= 5.0 (running instance)
- An OpenAI API key (for anchor resolution and fallback generation)

## License

MIT
