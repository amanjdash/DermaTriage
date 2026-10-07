import ProbabilityChart from './ProbabilityChart.jsx';
import { getLesionInfo } from '../services/lesionInfo.js';

function percent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

export default function PredictionResult({ result }) {
  if (!result) return null;
  const entropyPercent = Math.max(0, Math.min(100, Number(result.uncertainty_normalized) * 100));

  const predictedInfo = getLesionInfo(result.prediction);
  const deterministicInfo = getLesionInfo(result.deterministic_prediction);

  return (
    <section className="results-section" aria-labelledby="results-title" aria-live="polite">
      <div className="results-title-row">
        <div>
          <span className="eyebrow">02 / MODEL OUTPUT</span>
          <h2 id="results-title">Analysis summary</h2>
        </div>
        <span className="result-tag">
          <span className="status-dot" />Complete
        </span>
      </div>

      <div className="results-grid">
        <article className="prediction-card card">
          <span className="eyebrow">PREDICTED CLASS</span>
          <div className="class-row">
            <span className="class-symbol" aria-hidden="true">
              {predictedInfo.code.slice(0, 3).toUpperCase()}
            </span>
            <div className="class-title-group">
              <h3>{predictedInfo.friendlyName}</h3>
              <div className="class-meta-badges">
                <span className={`badge badge--${predictedInfo.badgeVariant}`}>
                  {predictedInfo.category}
                </span>
                <span className="code-pill">Code: {predictedInfo.code}</span>
              </div>
            </div>
          </div>
          <p className="result-description">
            Highest mean probability across {result.mc_passes} stochastic Monte Carlo dropout passes.
          </p>
          <div className="result-divider" />
          <div className="metric-row">
            <span>Model confidence</span>
            <strong>{percent(result.confidence)}</strong>
          </div>
          <div className="metric-row">
            <span>Deterministic pass</span>
            <strong>{deterministicInfo.shortLabel} ({result.deterministic_prediction})</strong>
          </div>
          <div className="metric-row">
            <span>Inference latency</span>
            <strong>{Number(result.inference_time_ms).toFixed(0)} ms</strong>
          </div>
        </article>

        <article className="uncertainty-card card">
          <div className="section-heading section-heading--compact">
            <div>
              <span className="eyebrow">EPISTEMIC ESTIMATE</span>
              <h2>Model uncertainty</h2>
            </div>
            <span
              className="info-mark"
              title="Predictive entropy measures spread across the model's mean class probabilities."
            >
              i
            </span>
          </div>
          <div className="uncertainty-value">
            {entropyPercent.toFixed(0)}
            <span>%</span>
          </div>
          <div
            className="uncertainty-track"
            role="meter"
            aria-label="Normalized predictive entropy"
            aria-valuemin="0"
            aria-valuemax="100"
            aria-valuenow={Math.round(entropyPercent)}
          >
            <span style={{ width: `${entropyPercent}%` }} />
          </div>
          <p>Normalized predictive entropy · raw entropy {Number(result.uncertainty).toFixed(3)} nats</p>
          <p className="uncertainty-note">
            Higher values indicate a less concentrated probability distribution. This is model uncertainty, not medical severity.
          </p>
        </article>

        <article className="lesion-info-card card" aria-labelledby="lesion-info-title">
          <div className="section-heading section-heading--compact">
            <div>
              <span className="eyebrow">CLINICAL GUIDE · TOP PREDICTION</span>
              <h2 id="lesion-info-title">Understanding {predictedInfo.shortLabel}</h2>
            </div>
            <span className={`badge badge--${predictedInfo.badgeVariant}`}>
              {predictedInfo.riskLevel}
            </span>
          </div>

          <p className="lesion-overview-lead">{predictedInfo.overview}</p>

          <div className="lesion-details-grid">
            <div className="lesion-detail-item">
              <span className="detail-label">Pathological Nature</span>
              <p>{predictedInfo.clinicalSignificance}</p>
            </div>
            <div className="lesion-detail-item">
              <span className="detail-label">Visual Characteristics</span>
              <p>{predictedInfo.visualAppearance}</p>
            </div>
            <div className="lesion-detail-item lesion-detail-item--action">
              <span className="detail-label">Recommended Next Steps</span>
              <p>{predictedInfo.recommendedAction}</p>
            </div>
          </div>

          <div className="medical-disclaimer-banner">
            <span className="disclaimer-icon" aria-hidden="true">⚠️</span>
            <p>
              <strong>Important medical notice:</strong> DermaTriage is an assistive AI research tool and does not provide an official clinical diagnosis. Any skin lesion causing concern, changing in appearance, or exhibiting symptoms should be evaluated in person by a board-certified dermatologist.
            </p>
          </div>
        </article>

        <ProbabilityChart probabilities={result.probabilities} />
      </div>
      <p className="model-stamp">{result.model_name} <span>·</span> version {result.model_version} <span>·</span> {result.device}</p>
    </section>
  );
}
