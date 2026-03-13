"""
HGCacheMem Graph Validator — the core novel contribution.

A topological Graph Validator that checks potential cache hits against
the user's active context within a knowledge graph prior to delivery,
improving contextual coherence and retrieval speed in agentic systems.
"""

from __future__ import annotations

from langchain_community.graphs import Neo4jGraph


class GraphValidator:
    """
    Validates semantic cache hits against a knowledge graph.

    The Graph Validator checks whether a candidate entity (returned by vector
    similarity search) is reachable from the user's active context anchor
    via a short weighted path in Neo4j.  Only paths where every edge
    weight exceeds ``weight_threshold`` are accepted.

    Parameters
    ----------
    graph : Neo4jGraph
        A connected LangChain Neo4jGraph instance.
    weight_threshold : float, optional
        Minimum edge weight required on every relationship in the path.
        Defaults to 0.6.
    max_hops : int, optional
        Maximum path length (number of relationships) for the shortest
        path check.  Defaults to 2.
    """

    def __init__(
        self,
        graph: Neo4jGraph,
        weight_threshold: float = 0.6,
        max_hops: int = 2,
    ) -> None:
        self.graph = graph
        self.weight_threshold = weight_threshold
        self.max_hops = max_hops

    def validate(self, active_entity: str, candidate_entity: str) -> bool:
        """
        Check whether *candidate_entity* is contextually valid given the
        user's current *active_entity* anchor.

        Returns ``True`` when:
        - The candidate **is** the active entity (identity case), or
        - A shortest path of at most ``max_hops`` exists between the two
          nodes and **every** relationship on that path has
          ``weight > weight_threshold``.

        Parameters
        ----------
        active_entity : str
            The ``name`` property of the user's current focus node.
        candidate_entity : str
            The ``name`` property of the node linked to the cache hit.

        Returns
        -------
        bool
        """
        if active_entity == candidate_entity:
            return True

        query = f"""
        MATCH (start {{name: $start_name}}), (end {{name: $end_name}})
        MATCH p = shortestPath((start)-[*..{self.max_hops}]-(end))
        WHERE all(r in relationships(p) WHERE r.weight > $threshold)
        RETURN p
        """

        result = self.graph.query(
            query,
            params={
                "start_name": active_entity,
                "end_name": candidate_entity,
                "threshold": self.weight_threshold,
            },
        )
        return bool(result)
