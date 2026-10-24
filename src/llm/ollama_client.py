"""
Client LLM Local Ollama pour Qwen 2.5 (3B).

Ce module encapsule les interactions avec le serveur local Ollama pour :
1. Décomposition d'affirmations scientifiques et extraction de mots-clés (Agent Analyseur).
2. Reformulation active de requêtes en cas de feedback (Agent Analyseur - boucle de rétroaction).
3. Rédaction d'une synthèse analytique en langage naturel (Agent Synthèse).

Tolérance aux pannes :
-----------------------
Si l'instance Ollama est inaccessible ou ne répond pas dans le délai imparti,
le module bascule automatiquement sur les heuristiques déterministes sans interrompre
l'exécution du pipeline multi-agents.
"""

import re
import json
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional

from src.config import (
    OLLAMA_BASE_URL,
    DEFAULT_LLM_MODEL,
    LLM_TIMEOUT_SECONDS
)

logger = logging.getLogger(__name__)


class QwenLLMClient:
    """
    Client d'inférence pour le LLM local Qwen 2.5 via l'API REST d'Ollama.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_LLM_MODEL,
        timeout: int = LLM_TIMEOUT_SECONDS
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._is_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Vérifie si le démon local Ollama est actif et accessible."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    self._is_available = True
                    return True
        except Exception:
            self._is_available = False
            return False
        return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2
    ) -> Optional[str]:
        """
        Envoie une requête de génération de texte à Qwen via l'endpoint /api/generate d'Ollama.
        """
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_ctx": 4096
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            logger.warning(f"Ollama non disponible sur {self.base_url} ({e}). Bascule sur heuristique.")
            return None
        except Exception as e:
            logger.warning(f"Erreur lors de l'appel LLM Qwen : {e}. Bascule sur heuristique.")
            return None

    def decompose_claim(self, claim: str) -> Dict[str, Any]:
        """
        Agent Analyseur (Round 0) : Utilise Qwen pour extraire les concepts et reformuler en requêtes ciblées.
        """
        system_prompt = (
            "You are a specialized AI assistant for scientific literature analysis (arXiv computer science). "
            "Your task is to analyze a scientific claim and output a JSON object only. "
            "Format: {\"keywords\": [\"term1\", \"term2\", \"term3\"], \"sub_queries\": [\"query1\", \"query2\"], \"summary\": \"one short sentence\"}"
        )
        prompt = (
            f"Analyze this scientific claim for academic retrieval:\n"
            f"Claim: \"{claim}\"\n"
            f"Return JSON only without markdown or codeblocks:"
        )

        response = self.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.1)
        if response:
            try:
                # Nettoyage d'éventuels backticks markdown
                clean_json = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.MULTILINE).strip()
                data = json.loads(clean_json)
                if isinstance(data, dict) and "keywords" in data:
                    return data
            except Exception:
                pass

        # Fallback heuristique si Ollama n'est pas actif ou a produit un format inattendu
        tokens = [t for t in re.findall(r'[A-Za-z0-9_-]{2,}', claim) if len(t) > 2]
        return {
            "keywords": tokens[:5],
            "sub_queries": [claim],
            "summary": "Décomposition heuristique"
        }

    def reformulate_query(self, claim: str, feedback_reason: str) -> str:
        """
        Agent Analyseur (Round > 0) : Utilise Qwen pour reformuler chirurgicalement la requête.
        """
        system_prompt = (
            "You are a search query reformulation expert for arXiv research papers. "
            "Output only the single best reformulated search query, without explanations or quotes."
        )
        prompt = (
            f"Original claim: \"{claim}\"\n"
            f"Reason previous search failed: \"{feedback_reason}\"\n"
            f"Provide an improved, targeted scientific query with key technical terms:"
        )

        response = self.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.3)
        if response and len(response.strip()) > 5:
            # Nettoyer les guillemets superflus
            return response.strip().strip('"').strip("'")

        # Fallback
        return f"{claim} technical empirical analysis"

    def generate_executive_summary(
        self,
        claim: str,
        verdict: str,
        verdict_fr: str,
        evidences: List[Dict[str, Any]],
        confidence: float
    ) -> str:
        """
        Agent Synthèse : Rédige une synthèse analytique en français appuyée strictement sur les preuves.
        """
        system_prompt = (
            "Tu es un expert senior en fact-checking scientifique et relecture par les pairs. "
            "Rédige une synthèse officielle et rigoureuse en français (2 à 4 phrases maximum). "
            "Règles impératives : "
            "1. Base ton analyse STRICTEMENT sur les extraits vérifiés fournis. "
            "2. N'invente aucun fait, aucun chiffre, aucun article extérieur (zéro hallucination). "
            "3. Explique avec clarté pourquoi la littérature soutient, réfute ou débat de l'affirmation."
        )

        # Concaténer les extraits des papiers retenus
        excerpts_text = ""
        for i, ev in enumerate(evidences[:3], 1):
            title = ev.get("title", "")
            raw = ev.get("raw_content", "")[:250]
            stance = ev.get("nli_stance", {}).get("label", "NEUTRAL")
            excerpts_text += f"\n[Extrait #{i} - {stance}] \"{title}\" : \"{raw}...\"\n"

        prompt = (
            f"Affirmation scientifique évaluée : \"{claim}\"\n"
            f"Verdict certifié : {verdict} ({verdict_fr})\n"
            f"Indice de certitude composite : {confidence * 100:.1f} %\n\n"
            f"Extraits scientifiques authentifiés issus d'arXiv :{excerpts_text}\n"
            f"Rédige la synthèse analytique officielle en français :"
        )

        response = self.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.2)
        if response and len(response.strip()) > 20:
            return response.strip()

        # Fallback textuel déterministe
        if verdict == "SOUTENU":
            return (
                f"Les publications scientifiques récentes confirment empiriquement l'affirmation avec un indice de certitude "
                f"de {confidence * 100:.1f} %. Les extraits analysés démontrent une convergence sans controverse notable."
            )
        elif verdict == "REFUTE":
            return (
                f"La littérature scientifique invalide l'affirmation avec un indice de certitude de {confidence * 100:.1f} %. "
                f"Les résultats expérimentaux répertoriés contredisent les postulats énoncés."
            )
        elif verdict == "SUJET_DEBATTU":
            return (
                f"L'affirmation fait l'objet de perspectives divergentes dans la littérature récente. "
                f"Certains articles valident l'hypothèse sous conditions restreintes tandis que d'autres établissent des contre-exemples."
            )
        else:
            return (
                f"Les preuves scientifiques indexées sont actuellement insuffisantes pour trancher avec certitude. "
                f"Conformément au principe d'honnêteté épistémique, aucun verdict binaire n'est imposé."
            )
