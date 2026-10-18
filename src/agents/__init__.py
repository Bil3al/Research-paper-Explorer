"""
Package multi-agents LangGraph pour la vérification scientifique autonome.

Expose l'état global du graphe (VerificationState) et les utilitaires de logging.
"""

from .state import (
    VerificationState,
    AgentLogEntry,
    create_initial_state,
    format_agent_log
)

__all__ = [
    "VerificationState",
    "AgentLogEntry",
    "create_initial_state",
    "format_agent_log"
]
