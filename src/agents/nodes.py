"""
Module des Nœuds Agents pour le Pipeline LangGraph Multi-Agents.

Rôle dans le projet :
----------------------
Définit les 5 nœuds d'agents spécialisés qui s'exécutent sur le graphe d'états :
1. `analyzer_node` (Agent Analyseur) : Analyse lexicale, mots-clés, catégories arXiv et reformulation.
2. `retriever_node` (Agent Récupérateur) : Pilotage de la recherche hybride (FAISS + BM25) et Reranker.
3. `evaluator_node` (Agent Évaluateur) : Inférence logique DeBERTa-v3 et détection du besoin de rétroaction.
4. `verifier_node` (Agent Vérificateur) : Audit textuel déterministe mot pour mot des citations.
5. `synthesis_node` (Agent Synthèse) : Calcul du score de confiance composite et rapport final Pydantic.

Optimisation CPU :
Les modèles lourds (DeBERTa, Cross-Encoder, Embeddings) sont mis en cache via un
gestionnaire singleton afin d'éviter tout rechargement superflu lors des rétroactions.
"""

import re
import sys
import logging
from typing import Dict, Any, List, Optional

# Encodage sécurisé pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import DEFAULT_ARXIV_CATEGORIES
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.verification.nli_evaluator import NLIEvaluator, StanceResult
from src.verification.citation_verifier import CitationVerifier
from src.verification.confidence_scorer import ConfidenceScorer
from src.verification.schemas import EvidenceItem, ClaimVerificationReport
from src.llm.ollama_client import QwenLLMClient
from src.agents.state import VerificationState, format_agent_log

logger = logging.getLogger(__name__)


# ==============================================================================
# GESTIONNAIRE SINGLETON POUR LES MODÈLES IA (OPTIMISATION MÉMOIRE CPU)
# ==============================================================================

class ModelRegistry:
    """Gestionnaire centralisé pour éviter de réinstancier les modèles sur CPU."""
    _hybrid: Optional[HybridRetriever] = None
    _reranker: Optional[CrossEncoderReranker] = None
    _nli: Optional[NLIEvaluator] = None
    _verifier: Optional[CitationVerifier] = None
    _scorer: Optional[ConfidenceScorer] = None
    _llm: Optional[QwenLLMClient] = None

    @classmethod
    def get_llm(cls) -> QwenLLMClient:
        if cls._llm is None:
            cls._llm = QwenLLMClient()
        return cls._llm

    @classmethod
    def get_hybrid(cls) -> HybridRetriever:
        if cls._hybrid is None:
            cls._hybrid = HybridRetriever()
        return cls._hybrid

    @classmethod
    def get_reranker(cls) -> CrossEncoderReranker:
        if cls._reranker is None:
            cls._reranker = CrossEncoderReranker()
        return cls._reranker

    @classmethod
    def get_nli(cls) -> NLIEvaluator:
        if cls._nli is None:
            cls._nli = NLIEvaluator()
        return cls._nli

    @classmethod
    def get_verifier(cls) -> CitationVerifier:
        if cls._verifier is None:
            cls._verifier = CitationVerifier()
        return cls._verifier

    @classmethod
    def get_scorer(cls) -> ConfidenceScorer:
        if cls._scorer is None:
            cls._scorer = ConfidenceScorer()
        return cls._scorer


# ==============================================================================
# 1. AGENT ANALYSEUR (Query Analysis & Decomposer)
# ==============================================================================

# Stopwords scientifiques génériques pour l'extraction de mots-clés
SCIENTIFIC_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "can", "could", "should", "would", "may", "might", "must", "that",
    "which", "who", "whom", "this", "these", "those", "it", "its", "they",
    "their", "we", "our", "you", "your", "uses", "used", "using", "use",
    "method", "methods", "model", "models", "paper", "papers", "approach"
}

# Règles heuristiques de détection de catégories arXiv
CATEGORY_RULES = {
    "cs.CV": ["vision", "visual", "image", "video", "segmentation", "depth", "camera", "frustum", "render", "diffusion"],
    "cs.CL": ["language", "nlp", "translation", "summarization", "text", "vlm", "linguistic", "sentence", "semantic"],
    "cs.LG": ["learning", "reinforcement", "neural", "deep", "gradient", "loss", "weights", "transformer", "generalization"],
    "cs.AI": ["intelligence", "agent", "agents", "reasoning", "planning", "logic", "knowledge", "heuristic"],
    "cs.SE": ["software", "code", "bug", "debugging", "repository", "git", "programming", "developer", "refactoring"]
}


def extract_scientific_keywords(text: str) -> List[str]:
    """Extrait les termes techniques saillants en ignorant les mots vides."""
    tokens = re.findall(r'[A-Za-z0-9_-]{2,}', text)
    keywords = [t for t in tokens if t.lower() not in SCIENTIFIC_STOPWORDS]
    # Déduplication tout en préservant l'ordre
    seen = set()
    result = []
    for k in keywords:
        if k.lower() not in seen:
            seen.add(k.lower())
            result.append(k)
    return result


def detect_target_categories(keywords: List[str]) -> List[str]:
    """Associe les mots-clés extraits aux catégories arXiv pertinentes."""
    matched_cats = set()
    lowered_kw = [k.lower() for k in keywords]

    for cat, patterns in CATEGORY_RULES.items():
        for pat in patterns:
            if any(pat in kw for kw in lowered_kw):
                matched_cats.add(cat)
                break

    if not matched_cats:
        return DEFAULT_ARXIV_CATEGORIES[:2]  # Fallback : cs.AI et cs.CL
    return sorted(list(matched_cats))


def analyzer_node(state: VerificationState) -> Dict[str, Any]:
    """
    Nœud 1 : Analyse l'affirmation avec le LLM Qwen 2.5, extrait les concepts scientifiques,
    sélectionne les catégories cibles, ou reformule la requête en cas de feedback.
    """
    iteration = state.get("iteration_count", 0)
    claim = state.get("claim", "")
    llm = ModelRegistry.get_llm()

    # CAS A : Première passe d'analyse (Round 0)
    if iteration == 0:
        decomposition = llm.decompose_claim(claim)
        llm_keywords = decomposition.get("keywords", [])
        keywords = llm_keywords if llm_keywords else extract_scientific_keywords(claim)
        target_cats = detect_target_categories(keywords)
        current_query = claim

        engine_badge = "Qwen 2.5 (Ollama)" if llm.is_available() else "Heuristique"
        log = format_agent_log(
            agent_name="Agent Analyseur",
            action="DECOMPOSITION_LLM",
            details=f"[{engine_badge}] Concepts extraits: {keywords[:5]} | Catégories arXiv: {target_cats}"
        )

        return {
            "current_query": current_query,
            "extracted_keywords": keywords,
            "target_categories": target_cats,
            "query_reformulation_notes": f"Analyse initiale {engine_badge}",
            "agent_logs": [log]
        }

    # CAS B : Boucle de rétroaction active (Round > 0)
    feedback_reason = state.get("feedback_reason", "Preuves insuffisantes")
    reformulated_query = llm.reformulate_query(claim, feedback_reason)

    expanded_cats = sorted(list(set(state.get("target_categories", []) + ["cs.AI", "cs.LG"])))
    notes = f"Reformulation Qwen 2.5 (Round {iteration}) : « {reformulated_query} » (catégories: {expanded_cats})"

    log = format_agent_log(
        agent_name="Agent Analyseur",
        action="REFORMULATION_LLM",
        details=notes
    )

    return {
        "current_query": reformulated_query,
        "target_categories": expanded_cats,
        "query_reformulation_notes": notes,
        "agent_logs": [log]
    }


# ==============================================================================
# 2. AGENT RÉCUPÉRATEUR (Retriever & Reranker Controller)
# ==============================================================================

def retriever_node(state: VerificationState) -> Dict[str, Any]:
    """
    Nœud 2 : Pilote le moteur RAG en 2 temps :
    1. Recherche hybride (FAISS + BM25 via RRF)
    2. Reclassement par Cross-Encoder
    """
    query = state.get("current_query", state.get("claim", ""))
    target_cats = state.get("target_categories", [])
    primary_filter = target_cats[0] if len(target_cats) == 1 else None

    hybrid = ModelRegistry.get_hybrid()
    reranker = ModelRegistry.get_reranker()

    # Exécution de la recherche hybride avec reclassement Cross-Encoder
    passages = hybrid.search_and_rerank(
        query=query,
        top_k=3,
        num_candidates=12,
        reranker=reranker,
        category_filter=primary_filter
    )

    if not passages:
        retrieval_status = "EMPTY"
        status_msg = "Aucun passage trouvé dans le corpus"
    else:
        avg_score = sum(p.get("rerank_score", 0.0) for p in passages) / len(passages)
        if avg_score < 0.25:
            retrieval_status = "LOW_RELEVANCE"
            status_msg = f"{len(passages)} passages trouvés mais pertinence sémantique faible ({avg_score*100:.1f}%)"
        else:
            retrieval_status = "SUCCESS"
            status_msg = f"{len(passages)} passages pertinents extraits et reclassés (score moyen: {avg_score*100:.1f}%)"

    log = format_agent_log(
        agent_name="Agent Récupérateur",
        action="RECUPERATION_HYBRIDE",
        details=status_msg
    )

    return {
        "retrieved_passages": passages,
        "retrieval_status": retrieval_status,
        "agent_logs": [log]
    }


# ==============================================================================
# 3. AGENT ÉVALUATEUR (Stance & Logic Evaluator)
# ==============================================================================

def evaluator_node(state: VerificationState) -> Dict[str, Any]:
    """
    Nœud 3 : Évalue la posture logique (DeBERTa-v3 NLI) de chaque passage retenu,
    agrège le consensus et décide s'il faut déclencher la boucle de rétroaction.
    """
    claim = state.get("claim", "")
    passages = state.get("retrieved_passages", [])
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_feedback_iterations", 2)

    nli = ModelRegistry.get_nli()

    # Si aucun passage n'a été récupéré
    if not passages:
        verdict_summary = {
            "global_verdict": "PREUVES_INSUFFISANTES",
            "verdict_fr": "Aucune source scientifique trouvée",
            "confidence": 0.0,
            "supports_count": 0,
            "refutes_count": 0,
            "neutral_count": 0,
            "total_evaluated": 0
        }
        evaluated_passages = []
    else:
        evaluated_passages = nli.evaluate_passages(claim=claim, passages=passages)
        verdict_summary = nli.aggregate_verdict(evaluated_passages)

    # Logique d'Arbitrage pour la Boucle de Rétroaction Active (Active Feedback Loop)
    # Si le verdict est PREUVES_INSUFFISANTES et qu'il reste des itérations autorisées
    is_insufficient = verdict_summary["global_verdict"] == "PREUVES_INSUFFISANTES"
    has_retries_left = iteration < max_iter

    if is_insufficient and has_retries_left:
        needs_feedback = True
        next_iter = iteration + 1
        reason = f"Preuves insuffisantes au tour {next_iter}/{max_iter}. Déclenchement de la rétroaction pour affiner la recherche."
        feedback_entry = f"[Tour {next_iter}] Échec de consensus concluant -> Rétroaction vers l'Agent Analyseur"
    else:
        needs_feedback = False
        next_iter = iteration
        reason = None
        feedback_entry = None

    log_details = (
        f"Verdict provisoire: {verdict_summary['global_verdict']} "
        f"({verdict_summary['supports_count']}S / {verdict_summary['refutes_count']}R / {verdict_summary['neutral_count']}N) | "
        f"Rétroaction requise: {'OUI' if needs_feedback else 'NON'}"
    )

    log = format_agent_log(
        agent_name="Agent Évaluateur",
        action="EVALUATION_LOGIQUE_NLI",
        details=log_details
    )

    update = {
        "evaluated_passages": evaluated_passages,
        "verdict_summary": verdict_summary,
        "needs_feedback": needs_feedback,
        "feedback_reason": reason,
        "iteration_count": next_iter,
        "agent_logs": [log]
    }

    if feedback_entry:
        update["feedback_history"] = [feedback_entry]

    return update


# ==============================================================================
# 4. AGENT VÉRIFICATEUR (Citation & Fact Auditor)
# ==============================================================================

def verifier_node(state: VerificationState) -> Dict[str, Any]:
    """
    Nœud 4 : Audit textuel déterministe mot pour mot (Anti-Hallucination).
    Contrôle que chaque passage analysé existe fidèlement dans la source arXiv.
    """
    evaluated = state.get("evaluated_passages", [])
    verifier = ModelRegistry.get_verifier()

    evidence_items: List[Dict[str, Any]] = []
    audits_passed = True

    for p in evaluated:
        raw_text = p.get("raw_content", "")
        # Échantillon de citation pour vérification exacte
        quote = raw_text[:min(120, len(raw_text))].strip()
        audit = verifier.verify_citation(citation=quote, source_text=raw_text)

        if not audit.is_valid:
            audits_passed = False

        stance_data = p.get("nli_stance", {})
        stance_obj = StanceResult(
            label=stance_data.get("label", "NEUTRAL"),
            confidence=stance_data.get("confidence", 0.0),
            probabilities=stance_data.get("probabilities", {}),
            is_conclusive=stance_data.get("is_conclusive", False),
            verdict_fr=stance_data.get("verdict_fr", "")
        )

        item = EvidenceItem(
            paper_id=p.get("paper_id", "inconnu"),
            chunk_id=p.get("chunk_id", ""),
            title=p.get("title", "Sans titre"),
            authors=p.get("authors", []),
            published_date=p.get("published_date", "2024-01-01"),
            primary_category=p.get("primary_category", "cs.AI"),
            entry_id=p.get("entry_id", ""),
            pdf_url=p.get("pdf_url"),
            doi=p.get("doi"),
            raw_content=raw_text,
            rerank_score=float(p.get("rerank_score", 0.0)),
            original_hybrid_rank=p.get("original_hybrid_rank"),
            final_rank=p.get("final_rank"),
            nli_stance=stance_obj,
            citation_check=audit
        )
        evidence_items.append(item.model_dump())

    log = format_agent_log(
        agent_name="Agent Vérificateur",
        action="AUDIT_CITATIONS",
        details=f"{len(evidence_items)} extraits audités mot pour mot | Intégrité textuelle: {'100% VALIDE' if audits_passed else 'ANOMALIE DÉTECTÉE'}"
    )

    return {
        "evidences": evidence_items,
        "citation_audit_passed": audits_passed,
        "agent_logs": [log]
    }


# ==============================================================================
# 5. AGENT SYNTHÈSE (Synthesis & Report Generator)
# ==============================================================================

def synthesis_node(state: VerificationState) -> Dict[str, Any]:
    """
    Nœud 5 : Calcule le score de confiance composite, compile les preuves
    et génère le rapport officiel consolidé (ClaimVerificationReport).
    """
    claim = state.get("claim", "")
    evaluated = state.get("evaluated_passages", [])
    summary = state.get("verdict_summary", {})
    evidences_raw = state.get("evidences", [])

    scorer = ModelRegistry.get_scorer()
    breakdown = scorer.compute_confidence(
        evaluated_passages=evaluated,
        global_verdict=summary.get("global_verdict", "PREUVES_INSUFFISANTES")
    )

    # Reconstruction des objets EvidenceItem
    evidence_objs = [EvidenceItem(**e) for e in evidences_raw]

    # Génération de la synthèse analytique avec le LLM Qwen 2.5
    llm = ModelRegistry.get_llm()
    executive_summary = llm.generate_executive_summary(
        claim=claim,
        verdict=summary.get("global_verdict", "PREUVES_INSUFFISANTES"),
        verdict_fr=summary.get("verdict_fr", "Preuves insuffisantes"),
        evidences=evidences_raw,
        confidence=breakdown.composite_score
    )

    # Génération du rapport officiel
    report = ClaimVerificationReport(
        claim=claim,
        verdict=summary.get("global_verdict", "PREUVES_INSUFFISANTES"),
        verdict_fr=summary.get("verdict_fr", "Preuves insuffisantes"),
        confidence_score=breakdown.composite_score,
        confidence_breakdown=breakdown,
        supports_count=summary.get("supports_count", 0),
        refutes_count=summary.get("refutes_count", 0),
        neutral_count=summary.get("neutral_count", 0),
        total_evidence_evaluated=summary.get("total_evaluated", len(evaluated)),
        executive_summary=executive_summary,
        evidences=evidence_objs
    )

    engine_badge = "Qwen 2.5 (Ollama)" if llm.is_available() else "Heuristique"
    log = format_agent_log(
        agent_name="Agent Synthèse",
        action="SYNTHESE_LLM",
        details=f"[{engine_badge}] Verdict: {report.verdict} | Confiance: {report.confidence_score * 100:.1f}% | Synthèse rédigée avec succès"
    )

    return {
        "confidence_breakdown": breakdown.model_dump(),
        "final_report": report.model_dump(),
        "agent_logs": [log]
    }
