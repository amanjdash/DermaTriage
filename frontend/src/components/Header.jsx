export default function Header({ apiStatus }) {
  const statusLabel = {
    checking: 'Checking API',
    ready: 'API connected',
    unavailable: 'API unavailable',
    modelMissing: 'Model not loaded',
  }[apiStatus];

  return (
    <header className="site-header">
      <a className="brand" href="#top" aria-label="DermaTriage home">
        <span className="brand-mark" aria-hidden="true"><span /><span /><span /><span /></span>
        <span className="brand-name">derma<span>triage</span><small>RESEARCH TRIAGE</small></span>
      </a>
      <div className="header-right">
        <span className={`api-status api-status--${apiStatus}`} role="status">
          <span className="status-dot" />{statusLabel}
        </span>
        <a className="header-link" href="#about">About this model <span aria-hidden="true">↗</span></a>
      </div>
    </header>
  );
}

