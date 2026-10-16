"""
Configuration globale du projet S5 : Vérification de claims scientifiques.
Ce fichier centralise les chemins de dossiers, les paramètres de collecte et les constantes.
"""

import os
from pathlib import Path

# ==============================================================================
# 1. CHEMINS DU PROJET (Arborescence)
# ==============================================================================

# Racine du projet (dossier Projet-S5)
BASE_DIR = Path(__file__).resolve().parent.parent

# Dossier principal des données (ignoré par Git pour ne pas surcharger le dépôt)
DATA_DIR = BASE_DIR / "data"

# Sous-dossiers spécifiques
RAW_DATA_DIR = DATA_DIR / "raw"          # Métadonnées brutes en JSON/JSONL
PROCESSED_DATA_DIR = DATA_DIR / "processed" # Chunks de texte nettoyés
PDFS_DIR = DATA_DIR / "pdfs"             # PDFs téléchargés (si besoin)
VECTOR_DB_DIR = DATA_DIR / "chroma_db"   # Persistance de la base vectorielle ChromaDB

# Fichier par défaut où sont stockés les articles collectés
DEFAULT_METADATA_FILE = RAW_DATA_DIR / "arxiv_papers_metadata.jsonl"


# ==============================================================================
# 2. PARAMÈTRES POUR LA COLLECTE ARXIV
# ==============================================================================

# Catégories d'informatique ciblées pour notre corpus (arXiv cs.*)
DEFAULT_ARXIV_CATEGORIES = [
    "cs.AI",  # Intelligence Artificielle générale
    "cs.CL",  # Traitement du Langage Naturel (NLP)
    "cs.LG",  # Machine Learning & Apprentissage profond
    "cs.CV",  # Vision par ordinateur
    "cs.SE",  # Génie logiciel
]

# Délai de politesse (en secondes) entre les requêtes vers l'API arXiv
# Recommandation officielle d'arXiv : 3 secondes pour ne pas surcharger leurs serveurs
ARXIV_DELAY_SECONDS = 3.0

# Nombre d'articles par page de requête (taille de lot)
ARXIV_BATCH_SIZE = 50


# ==============================================================================
# 3. CRÉATION AUTOMATIQUE DES DOSSIERS
# ==============================================================================

def init_directories():
    """
    Crée automatiquement tous les répertoires nécessaires s'ils n'existent pas encore.
    """
    for folder in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PDFS_DIR, VECTOR_DB_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


# Initialiser les dossiers au chargement de la configuration
init_directories()
