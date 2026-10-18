"""
Module verification : détection de posture logique (NLI DeBERTa) et vérification des citations.
"""

from .nli_evaluator import NLIEvaluator, StanceResult
from .citation_verifier import CitationVerifier

__all__ = [
    "NLIEvaluator",
    "StanceResult",
    "CitationVerifier"
]
