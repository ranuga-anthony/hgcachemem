"""
HGCacheMem Session State — tracks the user's current context anchor.
"""

from __future__ import annotations


class SessionState:
    """
    Working memory for a single user session.

    Keeps track of which graph node (the *anchor*) the user is currently
    focused on so that the Graph Validator can validate cache hits against it.

    Parameters
    ----------
    user_id : str
        An identifier for the session / user.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.active_entity_id: str | None = None

    def update_anchor(self, new_entity_id: str) -> None:
        """Move the session focus to *new_entity_id*."""
        self.active_entity_id = new_entity_id
