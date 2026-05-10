import { COMPONENT_LABELS, neighborhoodLabelForZip } from "../constants.js";
import { priceStatus } from "../filters.js";
import RentBar from "./RentBar.jsx";

function ComponentBar({ k, score }) {
  const pct = Math.max(0, Math.min(1, Number(score) || 0));
  return (
    <div className="comp" title={`${COMPONENT_LABELS[k]}: ${pct.toFixed(2)}`}>
      {COMPONENT_LABELS[k]}
      <div className="bar">
        <div style={{ width: `${(pct * 100).toFixed(0)}%` }} />
      </div>
      <span className="pct">{(pct * 100).toFixed(0)}</span>
    </div>
  );
}

export default function ListingCard({ listing, componentKeys, onClick }) {
  const status = priceStatus(listing);
  const className = `listing-card status-${status.kind}${
    listing.flag_overpriced ? " flagged" : ""
  }`;

  const title =
    listing.title || listing.address || `${listing.bedrooms || 0}-bed in ${listing.zip_code}`;
  const subtitle = listing.address || `ZIP ${listing.zip_code}`;
  const neighborhood = neighborhoodLabelForZip(listing.zip_code);

  const bandLabel =
    listing.fair_rent_p25 != null && listing.fair_rent_p75 != null
      ? `$${Math.round(listing.fair_rent_p25).toLocaleString()} – $${Math.round(listing.fair_rent_p75).toLocaleString()}`
      : `~$${Math.round(listing.predicted_rent_usd * 0.9).toLocaleString()} – $${Math.round(listing.predicted_rent_usd * 1.1).toLocaleString()} (est.)`;

  return (
    <article
      id={`card-${listing.listing_id}`}
      className={className}
      onClick={() => onClick?.(listing.listing_id)}
    >
      <div className="card-head">
        <div className="title-block">
          {neighborhood ? (
            <p className="neighborhood" title={neighborhood}>
              {neighborhood}
            </p>
          ) : null}
          <p className="title" title={title}>
            {title}
          </p>
          <p className="subtitle" title={subtitle}>
            {subtitle}
          </p>
        </div>
        <div className="rank">#{listing.rank}</div>
      </div>

      <div className="composite-row">
        <div className="composite-block">
          <span className="score-label">Composite score</span>
          <span className="score-badge">
            <span className="num">{(listing.composite_score * 100).toFixed(0)}</span>
            <span className="denom">/100</span>
          </span>
        </div>
        <span className={`price-pill ${status.kind}`} title={status.tooltip}>
          {status.label}
        </span>
      </div>

      <RentBar listing={listing} status={status} />

      <div className="kv-row">
        <span className="label">Asking</span>
        <span>${Math.round(listing.actual_rent_usd).toLocaleString()}/mo</span>
      </div>
      <div className="kv-row">
        <span className="label">Predicted (fair)</span>
        <span>${Math.round(listing.predicted_rent_usd).toLocaleString()}/mo</span>
      </div>
      <div className="kv-row">
        <span className="label">Fair band p25–p75</span>
        <span>{bandLabel}</span>
      </div>

      <div className="components">
        {componentKeys.map((k) => (
          <ComponentBar key={k} k={k} score={listing.component_scores?.[k]} />
        ))}
      </div>

      <div className="card-foot">
        <span>
          {listing.bedrooms || 0} bed
          {listing.bathrooms ? ` · ${listing.bathrooms} bath` : ""} · ZIP {listing.zip_code}
        </span>
        {listing.url ? (
          <a
            href={listing.url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
          >
            view ↗
          </a>
        ) : null}
        {listing.flag_overpriced ? (
          <span className="flag-pill" title={listing.flag_reason || ""}>
            overpriced
          </span>
        ) : null}
      </div>
    </article>
  );
}
