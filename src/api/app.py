"""
Application API REST FastAPI pour la Plateforme de Vérification Scientifique Souveraine.

Fonctionnalités :
-----------------
1. `GET /api/health` : Vérification de santé et disponibilité des moteurs IA locaux.
2. `GET /api/stats` : Métriques du corpus scientifique (articles, chunks, catégories).
3. `POST /api/verify` : Vérification synchrone d'une affirmation scientifique via LangGraph.
4. `POST /api/verify/stream` : Streaming en temps réel (Server-Sent Events) de l'avancement des agents.
"""

import sys
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timezone

# Encodage sécurisé pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_METADATA_FILE,
    DEFAULT_CHUNKS_FILE,
    FAISS_INDEX_FILE,
    BM25_INDEX_FILE,
    DEFAULT_ARXIV_CATEGORIES
)
from src.agents import (
    verification_pipeline,
    create_initial_state,
    verify_claim_agentic
)

logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Plateforme de Vérification Scientifique RAG Multi-Agents",
    description="API REST souveraine pour la vérification de claims scientifiques par RAG Hybride et Agents LangGraph.",
    version="1.0.0"
)

# Configuration CORS pour autoriser le frontend React (localhost:5173, localhost:3000, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# MODÈLES PYDANTIC POUR LES REQUÊTES
# ==============================================================================

class VerificationRequest(BaseModel):
    """Corps de la requête de vérification d'un claim scientifique."""
    claim: str = Field(
        ...,
        min_length=5,
        max_length=500,
        examples=["Video world models retain visual memory of generated content"],
        description="L'affirmation scientifique soumise à examen formel."
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=3,
        description="Plafond maximal d'itérations pour la boucle de rétroaction active."
    )


# ==============================================================================
# ENDPOINTS REST
# ==============================================================================

@app.get("/api/health", summary="Statut de santé du service")
def health_check() -> Dict[str, Any]:
    """Retourne l'état de fonctionnement du serveur et confirme l'exécution souveraine 100% locale."""
    return {
        "status": "healthy",
        "service": "Scientific Claim Verification Engine",
        "mode": "100% Local CPU Sovereign (No Cloud APIs)",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/stats", summary="Statistiques de la base de données et du corpus")
def get_corpus_stats() -> Dict[str, Any]:
    """Calcule et renvoie les statistiques du corpus scientifique local indexé."""
    paper_count = 0
    chunk_count = 0

    if DEFAULT_METADATA_FILE.exists():
        with open(DEFAULT_METADATA_FILE, "r", encoding="utf-8") as f:
            paper_count = sum(1 for line in f if line.strip())

    if DEFAULT_CHUNKS_FILE.exists():
        with open(DEFAULT_CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunk_count = sum(1 for line in f if line.strip())

    return {
        "total_papers_indexed": paper_count,
        "total_chunks": chunk_count,
        "vector_index_ready": FAISS_INDEX_FILE.exists(),
        "bm25_index_ready": BM25_INDEX_FILE.exists(),
        "supported_categories": DEFAULT_ARXIV_CATEGORIES,
        "embedding_dimensions": 384,
        "hybrid_method": "Reciprocal Rank Fusion (RRF, k=60) + Cross-Encoder",
        "llm_model": "Qwen 2.5 3B (Ollama)"
    }


@app.post("/api/verify", summary="Vérifier une affirmation scientifique")
def verify_claim(request: VerificationRequest) -> Dict[str, Any]:
    """
    Exécute le pipeline complet des 5 agents LangGraph de façon synchrone
    et renvoie le rapport officiel complet avec décomposition de la confiance.
    """
    try:
        final_state = verify_claim_agentic(
            claim=request.claim,
            max_feedback_iterations=request.max_retries
        )

        final_report = final_state.get("final_report", {})

        return {
            "success": True,
            "claim": final_state.get("claim"),
            "verdict": final_report.get("verdict", "PREUVES_INSUFFISANTES"),
            "verdict_fr": final_report.get("verdict_fr", "Preuves insuffisantes"),
            "confidence_score": final_report.get("confidence_score", 0.0),
            "confidence_breakdown": final_state.get("confidence_breakdown"),
            "supports_count": final_report.get("supports_count", 0),
            "refutes_count": final_report.get("refutes_count", 0),
            "neutral_count": final_report.get("neutral_count", 0),
            "total_evidence_evaluated": final_report.get("total_evidence_evaluated", 0),
            "executive_summary": final_report.get("executive_summary"),
            "evidences": final_report.get("evidences", []),
            "agent_logs": final_state.get("agent_logs", []),
            "feedback_history": final_state.get("feedback_history", []),
            "current_query": final_state.get("current_query"),
            "iteration_count": final_state.get("iteration_count", 0) + 1,
            "generated_at": final_report.get("generated_at")
        }

    except Exception as e:
        logger.exception("Erreur lors de l'exécution du pipeline multi-agents")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")


@app.post("/api/verify/stream", summary="Vérifier une affirmation avec streaming temps réel (SSE)")
async def verify_claim_stream(request: VerificationRequest):
    """
    Exécute le pipeline LangGraph en streaming via Server-Sent Events (SSE).
    Émet un événement JSON à chaque fois qu'un agent termine son étape.
    """
    def event_generator():
        initial_state = create_initial_state(
            claim=request.claim,
            max_feedback_iterations=request.max_retries
        )

        accumulated_state = dict(initial_state)

        try:
            # Diffuser le démarrage initial
            yield f"data: {json.dumps({'type': 'START', 'claim': request.claim, 'timestamp': datetime.now(timezone.utc).isoformat()})}\n\n"

            # Stream des mises à jour nœud par nœud de LangGraph
            for step_chunk in verification_pipeline.stream(initial_state, stream_mode="updates"):
                for node_name, node_update in step_chunk.items():
                    # Fusionner dans l'état accumulé
                    accumulated_state.update(node_update)

                    step_payload = {
                        "type": "NODE_UPDATE",
                        "node": node_name,
                        "logs": node_update.get("agent_logs", []),
                        "needs_feedback": node_update.get("needs_feedback"),
                        "feedback_reason": node_update.get("feedback_reason"),
                        "current_query": node_update.get("current_query"),
                        "verdict_summary": node_update.get("verdict_summary")
                    }
                    yield f"data: {json.dumps(step_payload)}\n\n"

            # Diffuser le rapport final complet
            final_report = accumulated_state.get("final_report", {})
            done_payload = {
                "type": "DONE",
                "success": True,
                "claim": accumulated_state.get("claim"),
                "verdict": final_report.get("verdict"),
                "verdict_fr": final_report.get("verdict_fr"),
                "confidence_score": final_report.get("confidence_score"),
                "confidence_breakdown": accumulated_state.get("confidence_breakdown"),
                "executive_summary": final_report.get("executive_summary"),
                "evidences": final_report.get("evidences", []),
                "agent_logs": accumulated_state.get("agent_logs", []),
                "feedback_history": accumulated_state.get("feedback_history", []),
                "iteration_count": accumulated_state.get("iteration_count", 0) + 1,
                "generated_at": final_report.get("generated_at")
            }
            yield f"data: {json.dumps(done_payload)}\n\n"

        except Exception as err:
            err_payload = {"type": "ERROR", "error": str(err)}
            yield f"data: {json.dumps(err_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ==============================================================================
# MONTAGE DE L'APPLICATION REACT (FRONTEND WEB MVP)
# ==============================================================================

from fastapi.staticfiles import StaticFiles

FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

