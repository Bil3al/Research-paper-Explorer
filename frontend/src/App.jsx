import React, { useState, useEffect } from 'react';
import './App.css';
import {
  Search,
  CheckCircle,
  XCircle,
  AlertTriangle,
  HelpCircle,
  ExternalLink,
  ShieldCheck,
  Cpu,
  RefreshCw,
  Clock,
  Layers,
  ChevronDown,
  ChevronUp,
  Database,
  BookOpen
} from 'lucide-react';

const PRESET_CLAIMS = [
  {
    title: "🔬 Controverse Démontrée",
    claim: "Video world models retain visual memory of generated content",
    desc: "Détecte des sources contradictoires (Velox vs Memory as Weights)"
  },
  {
    title: "⚪ Preuves Insuffisantes",
    claim: "Velox uses spatial memory for video world models",
    desc: "Démontre l'honnêteté épistémique (refuse d'halluciner)"
  },
  {
    title: "🟢 Soutenu par la littérature",
    claim: "Diffusion models improve image synthesis",
    desc: "Recherche sur les modèles génératifs et vision"
  }
];

export default function App() {
  const [claim, setClaim] = useState(PRESET_CLAIMS[0].claim);
  const [maxRetries, setMaxRetries] = useState(2);
  const [loading, setLoading] = useState(false);
  const [activeStep, setActiveStep] = useState(0); // 0: idle, 1: analyseur, 2: recup, 3: eval, 4: verif, 5: synth
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const [showLogs, setShowLogs] = useState(true);
  const [showJson, setShowJson] = useState(false);

  // Charger les statistiques du corpus local au démarrage
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
    } catch (err) {
      console.warn("API FastAPI non disponible pour l'instant:", err);
    }
  };

  const handleVerify = async (claimToTest = claim) => {
    if (!claimToTest.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setActiveStep(1);

    // Simulation de progression visuelle des agents en attendant la réponse
    const timer1 = setTimeout(() => setActiveStep(2), 700);
    const timer2 = setTimeout(() => setActiveStep(3), 2000);
    const timer3 = setTimeout(() => setActiveStep(4), 3500);

    try {
      const response = await fetch('/api/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          claim: claimToTest,
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

  // Helper pour les badges de verdict
  const renderVerdictBadge = (verdict, text) => {
    switch (verdict) {
      case 'SOUTENU':
        return <span className="verdict-badge badge-soutenu"><CheckCircle size={22} /> {text}</span>;
      case 'REFUTE':
        return <span className="verdict-badge badge-refute"><XCircle size={22} /> {text}</span>;
      case 'SUJET_DEBATTU':
        return <span className="verdict-badge badge-debattu"><AlertTriangle size={22} /> {text}</span>;
      default:
        return <span className="verdict-badge badge-neutre"><HelpCircle size={22} /> {text}</span>;
    }
  };

  return (
    <div className="app-container">
      {/* HEADER */}
      <header className="header">
        <div className="brand-badge">
          <Cpu size={16} />
          <span>ARCHITECTURE 100% LOCALE CPU • ZERO CLOUD API</span>
        </div>
        <h1>Plateforme de Vérification Scientifique</h1>
        <p className="header-subtitle">
          Vérification souveraine de claims par RAG Hybride (FAISS + BM25), Inférence Logique DeBERTa-v3
          et Orchestration Multi-Agents LangGraph avec boucle de rétroaction active.
        </p>

        {stats && (
          <div className="tech-pills">
            <span className="tech-pill">📚 {stats.total_papers_indexed} Articles arXiv cs.*</span>
            <span className="tech-pill">🧩 {stats.total_chunks} Chunks Contextualisés</span>
            <span className="tech-pill">⚡ FAISS IndexFlatIP (384d)</span>
            <span className="tech-pill">🔍 BM25Okapi RRF (k=60)</span>
            <span className="tech-pill">🎯 Cross-Encoder MiniLM</span>
            <span className="tech-pill">🧠 DeBERTa-v3 NLI</span>
          </div>
        )}
      </header>

      {/* SEARCH CARD */}
      <section className="search-card">
        <div className="search-label">
          <span>Affirmation Scientifique à Vérifier (Claim)</span>
          <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Corpus arXiv CS indexé localement</span>
        </div>

        <div className="search-input-wrapper">
          <textarea
            className="claim-textarea"
            value={claim}
            onChange={(e) => setClaim(e.target.value)}
            placeholder="Ex: Video world models retain visual memory of generated content..."
            disabled={loading}
          />
        </div>

        <div className="search-controls">
          <div className="sample-claims">
            <span className="sample-title">Exemples :</span>
            {PRESET_CLAIMS.map((p, idx) => (
              <button
                key={idx}
                type="button"
                className="sample-btn"
                onClick={() => {
                  setClaim(p.claim);
                  handleVerify(p.claim);
                }}
                disabled={loading}
                title={p.desc}
              >
                {p.title}
              </button>
            ))}
          </div>

          <div className="action-group">
            <label style={{ fontSize: '0.85rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              Max Essais:
              <select
                className="retries-select"
                value={maxRetries}
                onChange={(e) => setMaxRetries(Number(e.target.value))}
                disabled={loading}
              >
                <option value={1}>1 tour</option>
                <option value={2}>2 tours (recommandé)</option>
                <option value={3}>3 tours</option>
              </select>
            </label>

            <button
              className="verify-btn"
              onClick={() => handleVerify()}
              disabled={loading || !claim.trim()}
            >
              {loading ? (
                <>
                  <div className="spinner"></div>
                  <span>Évaluation en cours...</span>
                </>
              ) : (
                <>
                  <Search size={18} />
                  <span>Vérifier formellement</span>
                </>
              )}
            </button>
          </div>
        </div>
      </section>

      {/* ERREUR */}
      {error && (
        <div className="feedback-banner" style={{ marginBottom: '2rem' }}>
          <AlertTriangle size={20} />
          <span>Erreur : {error}</span>
        </div>
      )}

      {/* TIMELINE DES AGENTS */}
      {(loading || result) && (
        <section className="timeline-card">
          <div className="timeline-title">
            <Layers size={18} color="#60a5fa" />
            <span>Orchestration du Graphe Multi-Agents LangGraph</span>
          </div>

          <div className="agents-stepper">
            <div className={`agent-step ${activeStep >= 1 ? (activeStep === 1 && loading ? 'active' : 'completed') : ''}`}>
              <div className="agent-header">
                <span>1. Analyseur</span>
              </div>
              <span className="agent-role">Mots-clés & Catégories</span>
            </div>

            <div className={`agent-step ${activeStep >= 2 ? (activeStep === 2 && loading ? 'active' : 'completed') : ''}`}>
              <div className="agent-header">
                <span>2. Récupérateur</span>
              </div>
              <span className="agent-role">FAISS + BM25 + Rerank</span>
            </div>

            <div className={`agent-step ${activeStep >= 3 ? (activeStep === 3 && loading ? 'active' : 'completed') : ''}`}>
              <div className="agent-header">
                <span>3. Évaluateur</span>
              </div>
              <span className="agent-role">DeBERTa NLI & Consensus</span>
            </div>

            <div className={`agent-step ${activeStep >= 4 ? (activeStep === 4 && loading ? 'active' : 'completed') : ''}`}>
              <div className="agent-header">
                <span>4. Vérificateur</span>
              </div>
              <span className="agent-role">Audit Mot pour Mot</span>
            </div>

            <div className={`agent-step ${activeStep >= 5 ? 'completed' : ''}`}>
              <div className="agent-header">
                <span>5. Synthèse</span>
              </div>
              <span className="agent-role">Score Composite & Rapport</span>
            </div>
          </div>

          {result && result.feedback_history && result.feedback_history.length > 0 && (
            <div className="feedback-banner">
              <RefreshCw size={18} />
              <div>
                <strong>Boucle de rétroaction active déclenchée :</strong>
                {result.feedback_history.map((fb, i) => (
                  <div key={i} style={{ fontSize: '0.8rem', marginTop: '0.2rem' }}>• {fb}</div>
                ))}
              </div>
            </div>
          )}
        </section>
      )}

      {/* RÉSULTAT FINAL */}
      {result && (
        <>
          {/* BANNIÈRE HERO DU VERDICT */}
          <section className={`verdict-hero ${result.verdict}`}>
            <div className="verdict-header-row">
              <div>
                <span style={{ fontSize: '0.85rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 700 }}>
                  Verdict Global Certifié
                </span>
                <div style={{ marginTop: '0.4rem' }}>
                  {renderVerdictBadge(result.verdict, result.verdict_fr)}
                </div>
              </div>

              <div className="confidence-gauge-box">
                <div className="gauge-value" style={{
                  color: result.confidence_score >= 0.75 ? '#34d399' : (result.confidence_score >= 0.55 ? '#fbbf24' : '#f87171')
                }}>
                  {(result.confidence_score * 100).toFixed(1)}%
                </div>
                <div className="gauge-label">
                  <strong>Indice de Certitude</strong>
                  <span>Score Épistémique Composite</span>
                </div>
              </div>
            </div>

            {/* DÉCOMPOSITION DU SCORE (4 DIMENSIONS) */}
            {result.confidence_breakdown && (
              <div>
                <div className="breakdown-grid">
                  <div className="metric-card">
                    <div className="metric-header">
                      <span>1. Consensus Multi-Sources (35%)</span>
                    </div>
                    <div className="metric-value">{(result.confidence_breakdown.score_consensus * 100).toFixed(1)}%</div>
                    <div className="metric-bar-bg">
                      <div className="metric-bar-fill" style={{
                        width: `${result.confidence_breakdown.score_consensus * 100}%`,
                        backgroundColor: '#3b82f6'
                      }}></div>
                    </div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-header">
                      <span>2. Pertinence Cross-Encoder (25%)</span>
                    </div>
                    <div className="metric-value">{(result.confidence_breakdown.score_reranker * 100).toFixed(1)}%</div>
                    <div className="metric-bar-bg">
                      <div className="metric-bar-fill" style={{
                        width: `${result.confidence_breakdown.score_reranker * 100}%`,
                        backgroundColor: '#8b5cf6'
                      }}></div>
                    </div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-header">
                      <span>3. Déduction DeBERTa (25%)</span>
                    </div>
                    <div className="metric-value">{(result.confidence_breakdown.score_nli * 100).toFixed(1)}%</div>
                    <div className="metric-bar-bg">
                      <div className="metric-bar-fill" style={{
                        width: `${result.confidence_breakdown.score_nli * 100}%`,
                        backgroundColor: '#10b981'
                      }}></div>
                    </div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-header">
                      <span>4. Récence Temporelle (15%)</span>
                    </div>
                    <div className="metric-value">{(result.confidence_breakdown.score_recency * 100).toFixed(1)}%</div>
                    <div className="metric-bar-bg">
                      <div className="metric-bar-fill" style={{
                        width: `${result.confidence_breakdown.score_recency * 100}%`,
                        backgroundColor: '#f59e0b'
                      }}></div>
                    </div>
                  </div>
                </div>

                <div className="breakdown-explanation">
                  <strong>Explication de la certitude :</strong> {result.confidence_breakdown.explanation}
                </div>
              </div>
            )}
          </section>

          {/* LISTE DES PREUVES */}
          <section className="evidence-section">
            <h2 className="section-title">
              <BookOpen size={22} color="#60a5fa" />
              <span>Preuves Scientifiques Validées ({result.evidences?.length || 0} Extraits)</span>
            </h2>

            <div className="evidence-cards-list">
              {result.evidences?.map((ev, i) => {
                const stance = ev.nli_stance || {};
                const label = stance.label;
                const pillClass = label === 'SUPPORTS' ? 'pill-supports' : (label === 'REFUTES' ? 'pill-refutes' : 'pill-neutral');
                const badgeLabel = label === 'SUPPORTS' ? '🟢 SOUTIENT' : (label === 'REFUTES' ? '🔴 CONTREDIT' : '⚪ NEUTRE');

                return (
                  <div key={i} className="evidence-card">
                    <div className="card-top">
                      <h3 className="card-paper-title">{ev.title}</h3>
                      <span className={`stance-pill ${pillClass}`}>
                        {badgeLabel} ({(stance.confidence * 100).toFixed(1)}% DeBERTa)
                      </span>
                    </div>

                    <div className="paper-meta-row">
                      <span className="paper-meta-item">
                        <Clock size={14} /> Publié : {ev.published_date}
                      </span>
                      <span className="paper-meta-item">
                        <Database size={14} /> Catégorie : {ev.primary_category}
                      </span>
                      <a href={ev.entry_id} target="_blank" rel="noopener noreferrer" className="paper-link paper-meta-item">
                        arXiv:{ev.paper_id} <ExternalLink size={12} />
                      </a>
                      {ev.pdf_url && (
                        <a href={ev.pdf_url} target="_blank" rel="noopener noreferrer" className="paper-link paper-meta-item">
                          PDF Direct <ExternalLink size={12} />
                        </a>
                      )}
                    </div>

                    <div className="quote-box">
                      "{ev.raw_content}"
                    </div>

                    <div className="card-footer">
                      <div className="audit-badge-valid">
                        <ShieldCheck size={16} />
                        <span>Audit anti-hallucination : 100% MOT POUR MOT (EXACT MATCH)</span>
                      </div>

                      <div className="scores-summary">
                        <span>Rerank: <strong>{(ev.rerank_score * 100).toFixed(1)}%</strong></span>
                        <span>Rang: #{ev.original_hybrid_rank} ➔ #{ev.final_rank}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          {/* JOURNAL D'AUDIT COMPLET (TIMELINE TABLE) */}
          <section className="logs-section">
            <button
              type="button"
              className="logs-toggle"
              onClick={() => setShowLogs(!showLogs)}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Clock size={18} color="#60a5fa" />
                <span>Journal d'Audit Séquentiel des Agents ({result.agent_logs?.length || 0} étapes)</span>
              </span>
              {showLogs ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
            </button>

            {showLogs && (
              <table className="logs-table">
                <thead>
                  <tr>
                    <th>Heure UTC</th>
                    <th>Agent</th>
                    <th>Action</th>
                    <th>Détails de l'étape</th>
                  </tr>
                </thead>
                <tbody>
                  {result.agent_logs?.map((log, idx) => (
                    <tr key={idx}>
                      <td style={{ fontFamily: 'monospace', color: '#64748b' }}>{log.timestamp}</td>
                      <td className="log-agent">{log.agent}</td>
                      <td className="log-action">{log.action}</td>
                      <td style={{ color: '#cbd5e1' }}>{log.details}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>

          {/* INSPECTEUR JSON CONFORME PYDANTIC */}
          <section className="logs-section" style={{ marginTop: '1.5rem' }}>
            <button
              type="button"
              className="logs-toggle"
              onClick={() => setShowJson(!showJson)}
            >
              <span>Contrat de Données JSON (ClaimVerificationReport Pydantic)</span>
              {showJson ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
            </button>

            {showJson && (
              <pre style={{
                background: '#090d16',
                padding: '1rem',
                borderRadius: '8px',
                marginTop: '1rem',
                fontSize: '0.82rem',
                overflowX: 'auto',
                color: '#38bdf8'
              }}>
                {JSON.stringify(result, null, 2)}
              </pre>
            )}
          </section>
        </>
      )}
    </div>
  );
}
