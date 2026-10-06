import ProbabilityChart from './ProbabilityChart.jsx';

function percent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

export default function PredictionResult({ result }) {
  if (!result) return null;
  const entropyPercent = Math.max(0, Math.min(100, Number(result.uncertainty_normalized) * 100));

  return (
    <section className="results-section" aria-labelledby="results-title" aria-live="polite">
      <div className="results-title-row"><div><span className="eyebrow">02 / MODEL OUTPUT</span><h2 id="results-title">Analysis summary</h2></div><span className="result-tag"><span className="status-dot" />Complete</span></div>
      <div className="results-grid">
        <article className="prediction-card card">
          <span className="eyebrow">PREDICTED CLASS</span>
          <div className="class-row"><span className="class-symbol" aria-hidden="true">{result.prediction?.slice(0, 2).toUpperCase()}</span><h3>{result.prediction}</h3></div>
          <p className="result-description">Highest mean probability across {result.mc_passes} stochastic model passes.</p>
          <div className="result-divider" />
          <div className="metric-row"><span>Model confidence</span><strong>{percent(result.confidence)}</strong></div>
          <div className="metric-row"><span>Deterministic pass</span><strong>{result.deterministic_prediction}</strong></div>
          <div className="metric-row"><span>Inference time</span><strong>{Number(result.inference_time_ms).toFixed(0)} ms</strong></div>
        </article>
        <article className="uncertainty-card card">
          <div className="section-heading section-heading--compact"><div><span className="eyebrow">EPISTEMIC ESTIMATE</span><h2>Model uncertainty</h2></div><span className="info-mark" title="Predictive entropy measures spread across the model's mean class probabilities.">i</span></div>
          <div className="uncertainty-value">{entropyPercent.toFixed(0)}<span>%</span></div>
          <div className="uncertainty-track" role="meter" aria-label="Normalized predictive entropy" aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(entropyPercent)}><span style={{ width: `${entropyPercent}%` }} /></div>
          <p>Normalized predictive entropy · raw entropy {Number(result.uncertainty).toFixed(3)} nats</p>
          <p className="uncertainty-note">Higher values indicate a less concentrated probability distribution. This is model uncertainty, not medical severity.</p>
        </article>
        <ProbabilityChart probabilities={result.probabilities} />
      </div>
      <p className="model-stamp">{result.model_name} <span>·</span> version {result.model_version} <span>·</span> {result.device}</p>
    </section>
  );
}

