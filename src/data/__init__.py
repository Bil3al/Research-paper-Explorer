"""
Module data : gestion de la collecte, de l'indexation et du prétraitement des articles.
"""

from .collector import ArxivCollector, PaperMetadata

__all__ = ["ArxivCollector", "PaperMetadata"]
