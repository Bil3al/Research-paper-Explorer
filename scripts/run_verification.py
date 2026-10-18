"""
Script de Démonstration de la Vérification Logique de Claims Scientifiques (DeBERTa NLI).

Ce script exécute le pipeline complet :
1. Recherche Hybride (FAISS + BM25 via RRF)
2. Reclassement de précision par Cross-Encoder
3. Inférence logique NLI avec DeBERTa-v3 sur chaque passage retenu
4. Agrégation du verdict global avec honnêteté épistémique

Exemples d'utilisation :
-------------------------
1. Vérifier une affirmation scientifique :
   python scripts/run_verification.py --claim "Video world models require visual memory of generated content"

2. Tester une affirmation contradictoire ou fausse :
   python scripts/run_verification.py --claim "Diffusion models cannot be used for continuous language generation"
"""

import sys
import argparse
from pathlib import Path

# Encodage sécurisé pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.verification.nli_evaluator import NLIEvaluator
from src.verification.citation_verifier import CitationVerifier


def main():
    parser = argparse.ArgumentParser(
        description="Vérification de posture logique NLI et contrôle des preuves scientifiques."
    )

    parser.add_argument(
        "--claim",
        type=str,
        default="Video world models retain visual memory of generated content",
        help="L'affirmation scientifique à vérifier formellement"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Nombre de passages de preuves à évaluer (par défaut: 3)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🔬 PLATEFORME DE VÉRIFICATION SCIENTIFIQUE — PIPELINE NLI DEBERTA")
    print("=" * 80)
    print(f"• Affirmation (Claim) : \"{args.claim}\"")
    print(f"• Passages analysés   : Top-{args.top_k}")
    print("=" * 80 + "\n")

    # Initialisation des composants
    print("[1/3] Initialisation du moteur de recherche hybride et du reranker...")
    hybrid = HybridRetriever()
    reranker = CrossEncoderReranker()

    print("[2/3] Chargement du modèle d'inférence logique DeBERTa-v3...")
    nli = NLIEvaluator()
    verifier = CitationVerifier()

    # Étape 1 : Recherche en 2 étapes (Hybride + Reranker)
    print(f"\n[3/3] Recherche des preuves dans le corpus arXiv...")
    ranked_passages = hybrid.search_and_rerank(
        query=args.claim,
        top_k=args.top_k,
        num_candidates=12,
        reranker=reranker
    )

    if not ranked_passages:
        print("[ERREUR] Aucun passage pertinent trouvé dans la base.")
        return

    print(f"[OK] {len(ranked_passages)} passages de haute pertinence sélectionnés.")
    print("-" * 80)

    # Étape 2 : Évaluation NLI sur chaque passage
    evaluated_passages = nli.evaluate_passages(claim=args.claim, passages=ranked_passages)

    # Étape 3 : Agrégation du Verdict Global
    summary = nli.aggregate_verdict(evaluated_passages)

    print("\n" + "=" * 80)
    print(f"⚖️ VERDICT GLOBAL : {summary['verdict_fr']}")
    print(f"• Niveau de certitude composite : {summary['confidence'] * 100:.1f}%")
    print(f"• Répartition des preuves       : {summary['supports_count']} Soutient | {summary['refutes_count']} Contredit | {summary['neutral_count']} Neutre")
    print("=" * 80)

    # Détail des preuves individuelles
    print("\n📑 DÉTAIL DES ARTICLES & ANALYSE LOGIQUE PAR DEBERTA :")
    print("=" * 80)

    for i, p in enumerate(evaluated_passages, start=1):
        stance = p["nli_stance"]
        label = stance["label"]

        if label == "SUPPORTS":
            badge = "🟢 SOUTIENT"
        elif label == "REFUTES":
            badge = "🔴 CONTREDIT"
        else:
            badge = "⚪ NEUTRE"

        print(f"\n--- PREUVE #{i} : {badge} (Confiance : {stance['confidence'] * 100:.1f}%) ---")
        print(f"• Titre de l'article : {p['title']}")
        print(f"• Papier ID          : {p['paper_id']} ({p['published_date']}) | Catégorie: {p['primary_category']}")
        print(f"• Lien arXiv         : {p['entry_id']}")
        print(f"• Pertinence Rerank  : {p['rerank_score'] * 100:.1f}% (Rang initial: #{p['original_hybrid_rank']})")
        print(f"• Distribution NLI   : Soutient={stance['probabilities']['SUPPORTS']:.1%} | Contredit={stance['probabilities']['REFUTES']:.1%} | Neutre={stance['probabilities']['NEUTRAL']:.1%}")
        print(f"• Extrait textuel analysé :")
        print(f"  \"{p['raw_content'][:320]}...\"")

        # Test du vérificateur textuel
        sample_quote = p['raw_content'][:80]
        check = verifier.verify_citation(sample_quote, p['raw_content'])
        print(f"• Audit textuel      : {'[VALIDE MOT POUR MOT]' if check.is_valid else '[INVALIDE]'}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
