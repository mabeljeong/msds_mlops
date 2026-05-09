function clamp(n) {
  return Math.max(2, Math.min(98, n));
}

export default function RentBar({ listing, status }) {
  const actual = Number(listing.actual_rent_usd) || 0;
  const predicted = Number(listing.predicted_rent_usd) || 0;
  const { p25, p75, hasBand } = status;

  // Bar spans both the band and the actual rent, with padding, so the marker
  // is always inside the visible bar regardless of whether the listing is
  // over/under.
  const lo = Math.min(actual, p25) * 0.9;
  const hi = Math.max(actual, p75) * 1.1;

  const pct = (v) => `${clamp(((v - lo) / (hi - lo)) * 100)}%`;
  const bandWidthPct = clamp(((p75 - p25) / (hi - lo)) * 100);
  const bandClass = hasBand ? "rent-band" : "rent-band synthetic";
  const bandTitle = hasBand
    ? `Fair band p25–p75: $${Math.round(p25)}–$${Math.round(p75)}`
    : `Estimated band (predicted ±10%): $${Math.round(p25)}–$${Math.round(p75)}`;

  return (
    <div
      className="rent-bar"
      title={`Actual $${Math.round(actual)} · Predicted $${Math.round(predicted)}${
        hasBand ? "" : " (placeholder)"
      }`}
    >
      <div className="rent-axis" />
      <div
        className={bandClass}
        style={{ left: pct(p25), width: `${bandWidthPct}%` }}
        title={bandTitle}
      />
      {hasBand ? (
        <div
          className="rent-line"
          style={{ left: pct(predicted) }}
          title={`Predicted: $${Math.round(predicted)}`}
        />
      ) : null}
      <div className={`rent-marker ${status.kind}`} style={{ left: pct(actual) }} />
      <div className="rent-tick" style={{ left: pct(p25) }}>
        ${(p25 / 1000).toFixed(1)}k
      </div>
      <div className="rent-tick" style={{ left: pct(p75) }}>
        ${(p75 / 1000).toFixed(1)}k
      </div>
      <div
        className="rent-tick actual-tick"
        style={{ left: pct(actual), bottom: 32 }}
      >
        ${Math.round(actual).toLocaleString()}
      </div>
    </div>
  );
}
