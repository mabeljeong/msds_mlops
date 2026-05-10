export default function TopBar({ apiStatus }) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <div className="brand">
          <span className="logo-dot" />
          <h1>RentRadar</h1>
        </div>
        <span className="tagline">Fair rent. On demand.</span>
      </div>
      <div className="topbar-right">
        <span className="chip chip-brand">SF Bay Area</span>
        <span className={`api-status ${apiStatus.kind}`}>{apiStatus.text}</span>
      </div>
    </header>
  );
}
