import { useEffect, useState } from 'react';
import Header from './components/Header.jsx';
import ImageUploader from './components/ImageUploader.jsx';
import PredictionResult from './components/PredictionResult.jsx';
import { analyzeImage, getHealth } from './services/api.js';

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let alive = true;
    getHealth()
      .then((health) => { if (alive) setApiStatus(health.model_loaded ? 'ready' : 'modelMissing'); })
      .catch(() => { if (alive) setApiStatus('unavailable'); });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    if (!file) { setPreviewUrl(''); return undefined; }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function handleAnalyze() {
    if (!file || loading) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const prediction = await analyzeImage(file);
      setResult(prediction);
      setApiStatus('ready');
    } catch (requestError) {
      setError(requestError.message || 'Analysis could not be completed.');
      if (requestError.message?.includes('API could not be reached')) setApiStatus('unavailable');
      if (requestError.message?.includes('checkpoint') || requestError.message?.includes('model')) setApiStatus('modelMissing');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div id="top" className="app-shell min-h-screen">
      <Header apiStatus={apiStatus} />
      <main className="page-content">
        <section className="hero-section">
          <div className="hero-copy">
            <span className="hero-kicker"><span className="hero-kicker-dot" />AI-ASSISTED RESEARCH PROTOTYPE</span>
            <h1>A clearer view.<br /><em>A careful first step.</em></h1>
            <p>Explore a dermoscopic image with a research model built to show its class probabilities and uncertainty alongside every prediction.</p>
          </div>
          <div className="hero-aside" aria-label="Model workflow"><span>HAM10000</span><i>→</i><span>EfficientNet-B3</span><i>→</i><span>MC Dropout</span></div>
        </section>

        <section className="analysis-panel" aria-label="Image analysis">
          <div className="analysis-main">
            <ImageUploader file={file} previewUrl={previewUrl} onFileChange={(nextFile) => { setFile(nextFile); setResult(null); setError(''); }} disabled={loading} />
            {previewUrl && <div className="preview-caption"><span className="preview-dot" />Image selected <span>·</span> Preview stays in this browser tab</div>}
          </div>
          <aside className="analyze-aside card">
            <span className="eyebrow">READY WHEN YOU ARE</span>
            <h2>Run an analysis</h2>
            <p>The model will return its top class, probability distribution, and estimated epistemic uncertainty.</p>
            <button className="button button--primary analyze-button" type="button" disabled={!file || loading} onClick={handleAnalyze}>
              {loading ? <><span className="spinner" aria-hidden="true" />Analyzing image…</> : <>Analyze image <span aria-hidden="true">↗</span></>}
            </button>
            <div className="analysis-meta"><span>7 classes</span><span>·</span><span>30 MC passes</span><span>·</span><span>CPU inference</span></div>
            {apiStatus === 'modelMissing' && <p className="inline-hint">Backend is reachable; a trained checkpoint has not been loaded.</p>}
          </aside>
        </section>

        {error && <div className="error-banner" role="alert"><span aria-hidden="true">!</span><div><strong>Analysis wasn’t completed</strong><p>{error}</p></div></div>}
        {loading && <div className="loading-note" role="status">Running a research model. This can take a little longer on CPU.</div>}
        <PredictionResult result={result} />

        {!result && <section className="empty-state" aria-label="Results appear here">
          <div className="empty-icon" aria-hidden="true"><svg viewBox="0 0 38 38" fill="none"><circle cx="17" cy="17" r="10" stroke="currentColor" strokeWidth="1.5"/><path d="m24.5 24.5 7 7M17 12v10M12 17h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg></div>
          <div><span className="eyebrow">YOUR RESULTS</span><h2>Class probabilities and uncertainty will appear here</h2><p>Choose an image to begin. No prediction is shown until the model returns a real response.</p></div>
          <span className="empty-index">02</span>
        </section>}

        <section id="about" className="about-grid">
          <article className="about-card" id="dataset"><span className="eyebrow">ABOUT THE MODEL</span><h2>Built for transparency</h2><p>DermaTriage reports the model’s predicted class, confidence, and predictive entropy. MC Dropout captures variation across repeated passes; it does not measure clinical risk.</p><a href="https://doi.org/10.7910/DVN/DBW86T" target="_blank" rel="noreferrer">HAM10000 dataset reference <span aria-hidden="true">↗</span></a></article>
          <article className="about-card about-card--muted"><span className="eyebrow">RESEARCH USE ONLY</span><h2>Not a diagnosis</h2><p>DermaTriage is an AI-assisted research/triage prototype and is not a substitute for professional medical diagnosis. Do not use it to make care decisions.</p><div className="limitation-tags"><span>Not clinically validated</span><span>Dataset bias</span><span>Domain shift</span></div></article>
        </section>
        <footer className="page-footer"><span>DermaTriage <i>·</i> educational research prototype</span><span>Uploaded images are not retained by this interface.</span></footer>
      </main>
    </div>
  );
}

