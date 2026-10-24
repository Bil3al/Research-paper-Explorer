"""
Module des Schémas Pydantic pour la Vérification Scientifique.

Rôle dans le projet :
----------------------
Définit les contrats de données stricts pour :
1. Chaque élément de preuve individuelle (EvidenceItem).
2. La décomposition explicative du score de confiance composite (ConfidenceBreakdown).
3. Le rapport final consolidé de vérification d'affirmation (ClaimVerificationReport).

Ces structures garantissent l'interopérabilité entre les modules de recherche,
les agents LangGraph (Phase 4), l'API REST FastAPI (Phase 5) et l'interface Web (Phase 6).
"""

import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Réutilisation des types déjà validés dans les modules NLI et Citation
from src.verification.nli_evaluator import StanceResult
from src.verification.citation_verifier import CitationCheckResult


class EvidenceItem(BaseModel):
    """
    Représente un passage scientifique retenu comme preuve formelle,
    avec ses métadonnées bibliographiques arXiv, ses scores de pertinence
    et son audit textuel déterministe anti-hallucination.
    """
    paper_id: str = Field(description="Identifiant arXiv de l'article (ex: '2401.12345')")
    chunk_id: str = Field(description="Identifiant unique du chunk textuel")
    title: str = Field(description="Titre de la publication scientifique")
    authors: List[str] = Field(default_factory=list, description="Liste des auteurs de la publication")
    published_date: str = Field(description="Date de publication de l'article (AAAA-MM-JJ)")
    primary_category: str = Field(description="Catégorie arXiv principale (ex: 'cs.AI', 'cs.CV')")
    entry_id: str = Field(description="URL de la page de l'article sur arXiv")
    pdf_url: Optional[str] = Field(default=None, description="Lien de téléchargement direct du PDF")
    doi: Optional[str] = Field(default=None, description="Identifiant DOI officiel si disponible")
    raw_content: str = Field(description="Extrait textuel brut issu de l'article")
    
    # Métriques de recherche et de reclassement
    rerank_score: float = Field(description="Score de pertinence sémantique issu du Cross-Encoder (0.0 à 1.0)")
    original_hybrid_rank: Optional[int] = Field(default=None, description="Rang initial dans le classement hybride RRF")
    final_rank: Optional[int] = Field(default=None, description="Rang final après reclassement de précision")
    
    # Inférence logique et vérification de citation
    nli_stance: StanceResult = Field(description="Verdict d'inférence logique DeBERTa-v3 (SUPPORTS, REFUTES, NEUTRAL)")
    citation_check: Optional[CitationCheckResult] = Field(default=None, description="Audit textuel anti-hallucination mot pour mot")


class ConfidenceBreakdown(BaseModel):
    """
    Décomposition mathématique et explicative des 4 dimensions
    du score de certitude épistémique composite.
    """
    score_consensus: float = Field(description="Score d'accord/consensus entre les sources scientifiques (0.0 à 1.0)")
    score_reranker: float = Field(description="Score moyen de pertinence sémantique Cross-Encoder (0.0 à 1.0)")
    score_nli: float = Field(description="Score moyen de certitude logique DeBERTa-v3 (0.0 à 1.0)")
    score_recency: float = Field(description="Score de récence temporelle basé sur la décroissance exponentielle (0.0 à 1.0)")
    weights: Dict[str, float] = Field(description="Pondérations appliquées à chaque dimension")
    composite_score: float = Field(description="Score global synthétique pondéré (0.0 à 1.0)")
    explanation: str = Field(description="Explication pédagogique de la justification du score")


class ClaimVerificationReport(BaseModel):
    """
    Rapport complet et certifié de vérification d'une affirmation scientifique.
    Format standardisé pour l'API, les agents autonomes et l'export JSON.
    """
    claim: str = Field(description="L'affirmation scientifique soumise à vérification")
    verdict: str = Field(description="Verdict standardisé : 'SOUTENU', 'REFUTE', 'SUJET_DEBATTU' ou 'PREUVES_INSUFFISANTES'")
    verdict_fr: str = Field(description="Libellé explicatif du verdict en français")
    confidence_score: float = Field(description="Score de certitude composite global (0.0 à 1.0)")
    confidence_breakdown: ConfidenceBreakdown = Field(description="Décomposition détaillée des dimensions du score")
    supports_count: int = Field(description="Nombre d'extraits concluant que l'affirmation est scientifiquement soutenue")
    refutes_count: int = Field(description="Nombre d'extraits concluant que l'affirmation est scientifiquement réfutée")
    neutral_count: int = Field(description="Nombre d'extraits neutres ou non concluants")
    total_evidence_evaluated: int = Field(description="Nombre total de passages évalués")
    executive_summary: Optional[str] = Field(default=None, description="Synthèse analytique rédigée par le LLM Qwen 2.5")
    evidences: List[EvidenceItem] = Field(description="Liste ordonnée des preuves textuelles avec métadonnées complètes")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Horodatage ISO 8601 de génération de la vérification"
    )
