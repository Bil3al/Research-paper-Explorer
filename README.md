# 🔬 VERITAS-CS : Plateforme Multi-Agents Souveraine de Vérification de Claims Scientifiques

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph%20Multi--Agents-orange.svg)](https://github.com/langchain-ai/langgraph)
[![VectorStore FAISS](https://img.shields.io/badge/VectorStore-FAISS%20(Meta%20AI)-blueviolet.svg)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/Lexical-BM25Okapi-yellow.svg)](https://github.com/dorianbrown/rank_bm25)
[![NLI DeBERTa](https://img.shields.io/badge/NLI-DeBERTa--v3-green.svg)](https://huggingface.co/cross-encoder/nli-deberta-v3-small)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61dafb.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**VERITAS-CS** (*Verifiable Inference & Transparent Agent System for Computer Science*) est une plateforme autonome et **100 % locale (CPU)** de vérification d'affirmations scientifiques (*claims*) en informatique (sections `cs.*` d'arXiv). Elle combine **recherche hybride à deux étages (Dense FAISS + Sparse BM25 + Cross-Encoder)**, **inférence logique formelle (DeBERTa-v3 NLI)**, **audit mot pour mot anti-hallucination** et une **orchestration multi-agents LangGraph à boucle de rétroaction active**.

---

## 📌 Pourquoi ce projet ?

Les modèles de langage génératifs conventionnels (LLMs) souffrent d'hallucinations critiques dans le domaine scientifique : ils inventent des faits, déforment des conclusions ou fabriquent de faux DOI / citations. 

**VERITAS-CS résout ce problème à la racine grâce à 3 principes fondamentaux :**
1. **Zéro hallucination sur les citations :** Chaque extrait cité est audité et validé **mot pour mot (*exact string match*)** dans l'article source arXiv.
2. **Honnêteté épistémique :** Le système refuse d'halluciner ; il sait conclure par *"Preuves insuffisantes"* ou *"Sujet débattu"* avec un indice de certitude chiffré.
3. **Souveraineté & Coût zéro (100 % Local CPU) :** Aucune donnée n'est envoyée à des API tierces fermées ou payantes. Tout fonctionne en local sur processeur standard.

---

## 🏗️ Architecture Multi-Agents (LangGraph)

Le cœur du système est modélisé sous la forme d'un graphe d'états cyclique **LangGraph** orchestrant **5 agents spécialisés** :

```text
                  ┌───────────────────────────────────────────┐
                  │          1. Agent Analyseur               │
                  │ Concepts clés, catégories arXiv & requêtes│
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │          2. Agent Récupérateur            │
                  │   FAISS (BGE) + BM25Okapi (Fusion RRF)    │
                  │      + Reclassement Cross-Encoder         │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │          3. Agent Évaluateur              │
                  │   Inférence DeBERTa-v3 (Stance NLI)       │
                  │  Consensus : Soutenu / Réfuté / Neutre    │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │          4. Agent Vérificateur            │
                  │  Audit textuel déterministe (Mot pour Mot)│
                  └─────────────────────┬─────────────────────┘
                                        │
              [Preuves insuffisantes ?] ──► (Boucle de rétroaction active vers Agent 1)
                                        │
                                        ▼ (Preuves certifiées)
                  ┌───────────────────────────────────────────┐
                  │          5. Agent Synthèse                │
                  │  Score de confiance composite & Rapport   │
                  └───────────────────────────────────────────┘
```

### Rôles des 5 Agents Spécialisés :
1. **Agent Analyseur :** Décompose l'affirmation scientifique, extrait les concepts clés et sélectionne les catégories arXiv pertinentes (`cs.AI`, `cs.LG`, `cs.CV`, `cs.CL`, `cs.RO`, `cs.SE`, `cs.CR`). En cas de retour de boucle, il reformule chirurgicalement la requête.
2. **Agent Récupérateur :** Moteur à deux étages. Étage 1 : recherche hybride combinant la sémantique vectorielle dense (FAISS + `bge-small-en-v1.5`) et la recherche lexicale exacte (BM25Okapi) via **Reciprocal Rank Fusion (RRF $k=60$)**. Étage 2 : reclassement de précision des meilleurs candidats via un **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`).
3. **Agent Évaluateur (Stance Detection) :** Évalue la relation logique entre chaque extrait et l'affirmation via **DeBERTa-v3 MNLI** (`SUPPORTS`, `REFUTES`, `NEUTRAL`). Il agrège le consensus épistémique et décide si la boucle active doit être déclenchée.
4. **Agent Vérificateur :** Contrôle textuel déterministe sans IA générative garantissant que 100 % des extraits cités figurent textuellement dans l'article arXiv source.
5. **Agent Synthèse :** Calcule le score de confiance composite pondéré (Consensus 35%, Reranker 25%, Logique NLI 25%, Récence 15%) et génère le rapport officiel validé par schéma **Pydantic**.

---

## 🛠️ Stack Technologique (100% Locale)

| Composant | Technologie | Justification |
| :--- | :--- | :--- |
| **Orchestration Multi-Agents** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Modélisation en graphe cyclique avec boucle conditionnelle active |
| **Embeddings Denses** | `BAAI/bge-small-en-v1.5` | Modèle SOTA compact (384d) optimisé pour les textes scientifiques |
| **Index Vectoriel** | [FAISS (Meta AI)](https://github.com/facebookresearch/faiss) | Indexation `IndexFlatIP` ultrarapide par produit scalaire normalisé L2 |
| **Recherche Lexicale** | `rank_bm25` (BM25Okapi) | Capture exacte des acronymes (`LoRA`, `DeBERTa`, etc.) et versions |
| **Fusion Hybride** | Reciprocal Rank Fusion (RRF) | Standard académique ($k=60$) pour combiner dense et sparse sans biais d'échelle |
| **Reranker** | Cross-Encoder `ms-marco-MiniLM-L-6-v2` | Attention croisée requête-document pour réordonner le Top-K passages |
| **Inférence Logique** | `cross-encoder/nli-deberta-v3-small` | Déduction NLI formelle sans risque d'hallucination de LLM génératif |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) | API REST asynchrone haute performance + documentation OpenAPI |
| **Interface Dashboard** | [React](https://react.dev/) + [Vite](https://vitejs.dev/) | Dashboard moderne, clair et sobre, avec affichage des agents et scores |

---

## 📊 État du Corpus Scientifique Indexé

Le corpus local est indexé sur **8 catégories majeures de l'informatique** :
* **Articles arXiv indexés :** `4 627 articles`
* **Passages contextualisés (Chunks) :** `11 432 chunks`
* **Vecteurs FAISS :** `11 432 vecteurs (384 dimensions)`
* **Catégories couvertes :** `cs.AI` (Intelligence Artificielle), `cs.LG` (Machine Learning), `cs.CV` (Vision par ordinateur), `cs.CL` (NLP & Langage), `cs.IR` (Recherche d'information & RAG), `cs.RO` (Robotique), `cs.SE` (Génie Logiciel), `cs.CR` (Cryptographie & Sécurité).

---

## 🚀 Installation & Démarrage Rapide

### Prérequis
* **Python 3.12**
* **Node.js** (v18+) & **npm** (pour le frontend React)

### 1. Cloner le projet
```bash
git clone https://github.com/Bil3al/Research-paper-Explorer.git
cd Research-paper-Explorer
```

### 2. Environnement virtuel Python
```powershell
# Sous Windows PowerShell :
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

# Installer les dépendances :
pip install -r requirements.txt
```

### 3. (Optionnel) Ré-exécuter le Scale-Up ou l'Indexation
```powershell
# Automatise la collecte, le chunking, FAISS et BM25 :
python scripts/scale_up_corpus.py --target 3000
```

### 4. Lancer l'Application (Backend FastAPI & Dashboard React)
```powershell
# Lancer le serveur unifié (API REST + Frontend servi) :
python scripts/run_api.py --port 8000
```

Accédez à :
* 🌐 **Dashboard Web :** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* 📚 **Documentation API (Swagger) :** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* 📊 **Métriques du Corpus :** [http://127.0.0.1:8000/api/stats](http://127.0.0.1:8000/api/stats)

---

## 💻 Exécution en Ligne de Commande (CLI)

Vous pouvez également exécuter le pipeline directement dans votre terminal :

```powershell
# Vérification complète avec le pipeline multi-agents :
python scripts/run_agentic_pipeline.py --claim "Video world models retain visual memory of generated content"

# Test comparatif du moteur hybride (Dense vs Sparse vs RRF) :
python scripts/run_hybrid_search.py --query "diffusion models video generation" --compare

# Explorateur interactif de la base de données :
python scripts/inspect_db.py
```

---

## 👥 Auteurs & Encadrement

* **Étudiants :** Bilal MALIH • Oussama MAHI
* **Encadrante :** Pr. Karima MOUMANE
* **Cadre académique :** Projet de Semestre 5 (S5) — Cursus Ingénieur / Master Informatique
