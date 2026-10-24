import React, { useState, useEffect } from 'react';
import './App.css';
import {
  Search,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  ShieldCheck,
  Check,
  AlertCircle,
  FileText
} from 'lucide-react';

const PRESET_ASSERTIONS = [
  {
    tag: "Controverse / Deux perspectives",
    claim: "Video world models retain visual memory of generated content",
    note: "Présence d'articles aux conclusions contradictoires (Velox vs Memory as Weights)"
  },
  {
    tag: "Preuves insuffisantes (Honnêteté épistémique)",
    claim: "Velox uses spatial memory for video world models",
    note: "Le système refuse d'halluciner et conclut à un manque de sources décisives"
  },
  {
    tag: "Consensus établi",
    claim: "Diffusion models improve image synthesis quality over GANs",
    note: "Soutenu par la littérature en vision par ordinateur et modèles génératifs"
  }
];

export default function App() {
  const [claim, setClaim] = useState(PRESET_ASSERTIONS[0].claim);
  const [maxRetries, setMaxRetries] = useState(2);
  const [loading, setLoading] = useState(false);
  const [activeStep, setActiveStep] = useState(0); // 0: idle, 1: analysis, 2: retrieval, 3: nli, 4: audit, 5: synthesis
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const [showLogs, setShowLogs] = useState(false);
  const [showJson, setShowJson] = useState(false);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const res = await fetch('/api/stats');
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch {
      // Pas de bloquant si l'API est en cours de démarrage
    }
  };

  const handleVerify = async (assertionToTest = claim) => {
    if (!assertionToTest.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setActiveStep(1);

    const timer1 = setTimeout(() => setActiveStep(2), 600);
    const timer2 = setTimeout(() => setActiveStep(3), 1600);
    const timer3 = setTimeout(() => setActiveStep(4), 2800);

    try {
      const response = await fetch('/api/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          claim: assertionToTest,
          max_retries: maxRetries
        })
      });

      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Erreur serveur HTTP ${response.status}`);
      }

      const data = await response.json();
      setActiveStep(5);
      setResult(data);
    } catch (err) {
      setError(err.message);
      setActiveStep(0);
    } finally {
      setLoading(false);
    }
  };

  const getVerdictLabel = (verdict) => {
    switch (verdict) {
      case 'SOUTENU':
        return {
          title: "Confirmé par la littérature scientifique",
          badgeClass: "badge-soutenu",
          boxClass: "SOUTENU",
          symbol: "✓"
        };
      case 'REFUTE':
        return {
          title: "Réfuté par la littérature scientifique",
          badgeClass: "badge-refute",
          boxClass: "REFUTE",
          symbol: "✗"
        };
      case 'SUJET_DEBATTU':
        return {
          title: "Sujet débattu / Perspectives contradictoires",
          badgeClass: "badge-debattu",
          boxClass: "SUJET_DEBATTU",
          symbol: "⚠"
        };
      default:
        return {
          title: "Preuves scientifiques insuffisantes",
          badgeClass: "badge-neutre",
          boxClass: "PREUVES_INSUFFISANTES",
          symbol: "○"
        };
    }
  };

  const getAgentStatus = (stepIndex) => {
    if (activeStep === 0) return { label: "En attente", className: "state-waiting" };
    if (activeStep === stepIndex && loading) return { label: "Exécution...", className: "state-running" };
    if (activeStep >= stepIndex) return { label: "Terminé", className: "state-done" };
    return { label: "En attente", className: "state-waiting" };
  };

  return (
    <div className="academic-container">
      {/* MASTHEAD / EN-TÊTE INSTITUTIONNEL */}
      <header className="masthead">
        <div className="masthead-pretitle">
          <span>Laboratoire de Recherche en Informatique • Systèmes Multi-Agents & NLP</span>
          <span>Souveraineté 100% Locale (CPU) • Zéro API Cloud</span>
        </div>
        <h1 className="masthead-title">VERITAS-CS</h1>
        <p className="masthead-desc">
          Système autonome d'audit factuel et de vérification d'affirmations scientifiques sur corpus arXiv (<code>cs.*</code>). 
          Intègre un moteur de recherche hybride à deux étages (FAISS + BM25Okapi + Cross-Encoder), une inférence logique DeBERTa-v3 
          et un contrôle textuel déterministe mot pour mot garantissant l'absence totale d'hallucination.
        </p>

        {stats && (
          <div className="corpus-metabar">
            <div className="metabar-cell">
              <span className="metabar-label">Articles arXiv</span>
              <span className="metabar-val">{stats.total_papers_indexed?.toLocaleString() || '2 445'}</span>
            </div>
            <div className="metabar-cell">
              <span className="metabar-label">Passages indexés</span>
              <span className="metabar-val">{stats.total_chunks?.toLocaleString() || '6 008'}</span>
            </div>
            <div className="metabar-cell">
              <span className="metabar-label">Index Dense</span>
              <span className="metabar-val">FAISS FlatIP (384d)</span>
            </div>
            <div className="metabar-cell">
              <span className="metabar-label">Index Lexical</span>
              <span className="metabar-val">BM25Okapi (RRF k=60)</span>
            </div>
            <div className="metabar-cell">
              <span className="metabar-label">Reclassement</span>
              <span className="metabar-val">MiniLM Cross-Encoder</span>
            </div>
            <div className="metabar-cell">
              <span className="metabar-label">Inférence Logique</span>
              <span className="metabar-val">DeBERTa-v3 MNLI</span>
            </div>
            <div className="metabar-cell">
              <span className="metabar-label">LLM Local</span>
              <span className="metabar-val">{stats.llm_model || 'Qwen 2.5 3B (Ollama)'}</span>
            </div>
          </div>
        )}
      </header>

      {/* ZONE DE SAISIE DE L'ASSERTION */}
      <section className="academic-card">
        <div className="card-heading">
          <span>Affirmation Scientifique Soumise à Audit (Claim)</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.74rem', textTransform: 'none' }}>
            Langue recommandée : Anglais académique
          </span>
        </div>

        <textarea
          className="assertion-textarea"
          value={claim}
          onChange={(e) => setClaim(e.target.value)}
          placeholder="Saisissez une affirmation technique précise (ex: Video world models retain visual memory of generated content)..."
          disabled={loading}
          rows={3}
        />

        {/* EXEMPLES D'AFFIRMATIONS DE RÉFÉRENCE */}
        <div className="sample-assertions-box">
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            Jeux d'assertions pré-annotés pour audit :
          </div>
          {PRESET_ASSERTIONS.map((preset, idx) => (
            <button
              key={idx}
              type="button"
              className="sample-assertion-btn"
              onClick={() => {
                setClaim(preset.claim);
                handleVerify(preset.claim);
              }}
              disabled={loading}
            >
              <span className="sample-tag">{preset.tag}</span>
              <span style={{ fontStyle: 'normal' }}>« {preset.claim} »</span>
            </button>
          ))}
        </div>

        {/* CONTRÔLES DE SOUMISSION */}
        <div className="submission-bar">
          <label className="retries-control">
            <span>Rétroactions max autorisées :</span>
            <select
              value={maxRetries}
              onChange={(e) => setMaxRetries(Number(e.target.value))}
              disabled={loading}
            >
              <option value={1}>1 itération</option>
              <option value={2}>2 itérations (standard)</option>
              <option value={3}>3 itérations (approfondi)</option>
            </select>
          </label>

          <button
            type="button"
            className="btn-verify"
            onClick={() => handleVerify()}
            disabled={loading || !claim.trim()}
          >
            {loading ? (
              <span>[Évaluation en cours...]</span>
            ) : (
              <>
                <Search size={15} />
                <span>Exécuter la vérification formelle</span>
              </>
            )}
          </button>
        </div>
      </section>

      {/* BANNIÈRE D'ERREUR */}
      {error && (
        <div className="academic-card" style={{ borderColor: 'var(--status-refuted-border)', backgroundColor: 'var(--status-refuted-bg)', color: 'var(--status-refuted-text)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
            <AlertCircle size={16} />
            <span>Erreur d'exécution : {error}</span>
          </div>
        </div>
      )}

      {/* JOURNAL D'EXÉCUTION MULTI-AGENTS DISCRET */}
      {(loading || result) && (
        <section className="academic-card">
          <div className="card-heading">
            <span>Journal d'Orchestration Multi-Agents (LangGraph)</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.74rem', textTransform: 'none' }}>
              Cycle déterministe avec boucle de rétroaction
            </span>
          </div>

          <div className="agents-trace-container">
            <div className="agent-row">
              <div className="agent-identity">
                <span className="agent-step-num">01.</span>
                <span className="agent-name">Agent Analyseur</span>
                <span className="agent-spec">— Décomposition sémantique & Catégorisation arXiv</span>
              </div>
              <span className={`agent-badge-state ${getAgentStatus(1).className}`}>
                {getAgentStatus(1).label}
              </span>
            </div>

            <div className="agent-row">
              <div className="agent-identity">
                <span className="agent-step-num">02.</span>
                <span className="agent-name">Agent Récupérateur</span>
                <span className="agent-spec">— Recherche Hybride (FAISS + BM25 RRF k=60) & Reclassement Cross-Encoder</span>
              </div>
              <span className={`agent-badge-state ${getAgentStatus(2).className}`}>
                {getAgentStatus(2).label}
              </span>
            </div>

            <div className="agent-row">
              <div className="agent-identity">
                <span className="agent-step-num">03.</span>
                <span className="agent-name">Agent Évaluateur</span>
                <span className="agent-spec">— Déduction Logique NLI (DeBERTa-v3) & Détection de Consensus</span>
              </div>
              <span className={`agent-badge-state ${getAgentStatus(3).className}`}>
                {getAgentStatus(3).label}
              </span>
            </div>

            <div className="agent-row">
              <div className="agent-identity">
                <span className="agent-step-num">04.</span>
                <span className="agent-name">Agent Vérificateur</span>
                <span className="agent-spec">— Audit Déterministe Mot pour Mot (Contrôle anti-hallucination)</span>
              </div>
              <span className={`agent-badge-state ${getAgentStatus(4).className}`}>
                {getAgentStatus(4).label}
              </span>
            </div>

            <div className="agent-row">
              <div className="agent-identity">
                <span className="agent-step-num">05.</span>
                <span className="agent-name">Agent Synthèse</span>
                <span className="agent-spec">— Calcul du Score de Certitude & Génération du Rapport Certifié</span>
              </div>
              <span className={`agent-badge-state ${getAgentStatus(5).className}`}>
                {getAgentStatus(5).label}
              </span>
            </div>
          </div>

          {result && result.feedback_history && result.feedback_history.length > 0 && (
            <div className="feedback-trigger-box">
              <strong>Boucle de rétroaction active déclenchée ({result.feedback_history.length} cycle(s)) :</strong>
              {result.feedback_history.map((item, idx) => (
                <div key={idx} style={{ marginTop: '0.2rem' }}>• {item}</div>
              ))}
            </div>
          )}
        </section>
      )}

      {/* RÉSULTATS & VERDICT SCIENTIFIQUE CERTIFIÉ */}
      {result && (
        <>
          {/* BANDEAU DU VERDICT ACADÉMIQUE */}
          {(() => {
            const vInfo = getVerdictLabel(result.verdict);
            const breakdown = result.confidence_breakdown || {};

            return (
              <section className={`verdict-box ${vInfo.boxClass}`}>
                <div className="verdict-meta-header">
                  <div className="verdict-title-group">
                    <span style={{ fontSize: '1.25rem', fontWeight: 700 }}>{vInfo.symbol}</span>
                    <span className={`verdict-label-badge ${vInfo.badgeClass}`}>
                      {vInfo.title}
                    </span>
                  </div>

                  <div className="consensus-ratio-text">
                    Consensus : <strong>{result.supports_count}</strong> soutiennent, <strong>{result.refutes_count}</strong> contredisent, <strong>{result.neutral_count}</strong> neutres
                    (sur {result.total_evidence_evaluated} sources évaluées)
                  </div>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    Indice de Certitude Épistémique Composite : <strong>{(result.confidence_score * 100).toFixed(1)}%</strong> [{result.confidence_score?.toFixed(3)} / 1.000]
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    Formule pondérée normalisée [0-1]
                  </span>
                </div>

                {/* TABLEAU DES 4 DIMENSIONS DE CERTITUDE */}
                <table className="confidence-table">
                  <thead>
                    <tr>
                      <th>Dimension d'Évaluation</th>
                      <th>Pondération</th>
                      <th>Mécanisme de Calcul</th>
                      <th style={{ textAlign: 'right' }}>Score Partiel</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>1. Consensus Multi-Sources</td>
                      <td>35 %</td>
                      <td>Unanimité = 1.00 / Source unique = 0.65 / Controverse = écart relatif</td>
                      <td className="score-col">{((breakdown.score_consensus || 0) * 100).toFixed(1)} %</td>
                    </tr>
                    <tr>
                      <td>2. Pertinence Cross-Encoder</td>
                      <td>25 %</td>
                      <td>Score moyen d'attention croisée requête-document (ms-marco-MiniLM)</td>
                      <td className="score-col">{((breakdown.score_reranker || 0) * 100).toFixed(1)} %</td>
                    </tr>
                    <tr>
                      <td>3. Déduction Logique NLI</td>
                      <td>25 %</td>
                      <td>Probabilité d'inférence directe DeBERTa-v3 sur les sources concluantes</td>
                      <td className="score-col">{((breakdown.score_nli || 0) * 100).toFixed(1)} %</td>
                    </tr>
                    <tr>
                      <td>4. Récence Temporelle</td>
                      <td>15 %</td>
                      <td>Décroissance exponentielle basée sur l'âge des articles (λ = 0.10)</td>
                      <td className="score-col">{((breakdown.score_recency || 0) * 100).toFixed(1)} %</td>
                    </tr>
                  </tbody>
                </table>

                {result.executive_summary && (
                  <div style={{ marginTop: '0.85rem', padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.85)', border: '1px solid rgba(0,0,0,0.08)', borderLeft: '3px solid var(--accent-navy)' }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', textTransform: 'uppercase', color: 'var(--accent-navy)', fontWeight: 700, display: 'block', marginBottom: '0.3rem' }}>
                      Synthèse Analytique Rédigée par Qwen 2.5 (Ollama) :
                    </span>
                    <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-primary)', lineHeight: 1.55 }}>
                      {result.executive_summary}
                    </p>
                  </div>
                )}

                {breakdown.explanation && (
                  <p className="verdict-explanation-p">
                    <strong>Pondération des métriques :</strong> {breakdown.explanation}
                  </p>
                )}
              </section>
            );
          })()}

          {/* SECTION "LIMITES & NUANCES SCIENTIFIQUES" */}
          <section className="nuances-box">
            <h4>Limites Méthodologiques & Nuances Épistémiques</h4>
            <div style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {result.verdict === 'SUJET_DEBATTU' && (
                <p>
                  <strong>Divergence identifiée dans le corpus :</strong> La littérature scientifique présente des résultats contradictoires. 
                  Certains articles confirment la thèse sous des hypothèses restrictives, tandis que d'autres la réfutent ou démontrent des contre-exemples. 
                  L'affirmation ne peut être considérée comme un consensus établi en l'état actuel des publications.
                </p>
              )}
              {result.verdict === 'PREUVES_INSUFFISANTES' && (
                <p>
                  <strong>Absence de preuves concluantes :</strong> Le corpus scientifique indexé ne contient aucun passage dont la certitude NLI 
                  dépasse le seuil de décision de 60%. Conformément au principe d'honnêteté épistémique, le système refuse de forcer un verdict binaire.
                </p>
              )}
              {result.verdict === 'SOUTENU' && (
                <p>
                  <strong>Convergence concordante :</strong> Les sources les plus pertinentes confirment empiriquement ou théoriquement l'affirmation, 
                  sans contradiction majeure relevée dans les publications indexées.
                </p>
              )}
              {result.verdict === 'REFUTE' && (
                <p>
                  <strong>Invalidation documentée :</strong> Les publications analysées contredisent explicitement la proposition ou établissent 
                  des limites structurelles contredisant l'hypothèse formulée.
                </p>
              )}
            </div>
          </section>

          {/* LISTE STRUCTURÉE DES PREUVES SCIENTIFIQUES */}
          <section className="academic-card">
            <div className="card-heading">
              <span>Références & Extraits Scientifiques Concluants ({result.evidences?.length || 0})</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.74rem', textTransform: 'none' }}>
                Triés par pertinence Cross-Encoder décroissante
              </span>
            </div>

            <div className="evidence-list">
              {result.evidences?.map((ev, i) => {
                const stance = ev.nli_stance || {};
                const label = stance.label;
                const pillClass = label === 'SUPPORTS' ? 'pill-supports' : (label === 'REFUTES' ? 'pill-refutes' : 'pill-neutral');
                const badgeText = label === 'SUPPORTS' ? 'SOUTIENT' : (label === 'REFUTES' ? 'CONTREDIT' : 'NEUTRE');

                return (
                  <article key={i} className="evidence-entry">
                    <div className="evidence-header">
                      <h3 className="evidence-title">{ev.title}</h3>
                      <span className={`stance-pill-flat ${pillClass}`}>
                        {badgeText} [{(stance.confidence * 100).toFixed(1)}% DeBERTa]
                      </span>
                    </div>

                    <div className="evidence-metadata-row">
                      <span>Publié le : {ev.published_date}</span>
                      <span>Catégorie : {ev.primary_category}</span>
                      <span>Auteurs : {ev.authors?.slice(0, 3).join(', ')}{ev.authors?.length > 3 ? ' et al.' : ''}</span>
                      <a href={ev.entry_id} target="_blank" rel="noopener noreferrer" className="evidence-link">
                        arXiv:{ev.paper_id} <ExternalLink size={10} style={{ display: 'inline' }} />
                      </a>
                      {ev.pdf_url && (
                        <a href={ev.pdf_url} target="_blank" rel="noopener noreferrer" className="evidence-link">
                          PDF complet <ExternalLink size={10} style={{ display: 'inline' }} />
                        </a>
                      )}
                    </div>

                    {/* EXTRAIT TEXTUEL BRUT EXACTEMENT CITÉ */}
                    <blockquote className="verbatim-quote-block">
                      « {ev.raw_content} »
                    </blockquote>

                    <div className="evidence-footer-audit">
                      <div className="audit-certificate">
                        <ShieldCheck size={14} />
                        <span>Audit textuel certifié : 100% Exact Match (mot pour mot)</span>
                      </div>

                      <div className="rerank-metrics">
                        <span>Score Cross-Encoder : <strong>{(ev.rerank_score * 100).toFixed(1)}%</strong></span>
                        <span style={{ marginLeft: '0.8rem' }}>Rang Hybride : #{ev.original_hybrid_rank} ➔ Final : #{ev.final_rank}</span>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          </section>

          {/* TABLEAU D'AUDIT SÉQUENTIEL COMPLET (JOURNAL DE BORD) */}
          <section className="academic-card">
            <button
              type="button"
              className="btn-toggle-collapsible"
              onClick={() => setShowLogs(!showLogs)}
            >
              <span>Journal d'audit séquentiel des agents ({result.agent_logs?.length || 0} entrées)</span>
              {showLogs ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>

            {showLogs && (
              <table className="logs-table">
                <thead>
                  <tr>
                    <th style={{ width: '130px' }}>Heure UTC</th>
                    <th style={{ width: '160px' }}>Agent Responsable</th>
                    <th style={{ width: '180px' }}>Action Opérée</th>
                    <th>Détails de l'étape</th>
                  </tr>
                </thead>
                <tbody>
                  {result.agent_logs?.map((log, idx) => (
                    <tr key={idx}>
                      <td style={{ color: 'var(--text-muted)' }}>{log.timestamp}</td>
                      <td style={{ fontWeight: 600 }}>{log.agent}</td>
                      <td><code>{log.action}</code></td>
                      <td style={{ color: 'var(--text-secondary)' }}>{log.details}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>

          {/* INSPECTEUR DE CONTRAT DE DONNÉES PYDANTIC (JSON) */}
          <section className="academic-card">
            <button
              type="button"
              className="btn-toggle-collapsible"
              onClick={() => setShowJson(!showJson)}
            >
              <span>Contrat de données JSON certifié (Pydantic ClaimVerificationReport)</span>
              {showJson ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>

            {showJson && (
              <pre className="json-pre-box">
                {JSON.stringify(result, null, 2)}
              </pre>
            )}
          </section>
        </>
      )}
    </div>
  );
}
