import { COMPONENT_LABELS } from "../constants.js";
import ListingCard from "./ListingCard.jsx";

export default function ResultsList({
  items,
  componentKeys,
  weightsNormalized,
  onCardClick,
}) {
  return (
    <>
      <div className="results-header">
        <h2 id="resultsTitle">
          {items.length ? `Top picks (${items.length})` : "No matches"}
        </h2>
        <div id="resultsMeta">
          {componentKeys.map((k) => (
            <span className="weight-chip" key={k}>
              {COMPONENT_LABELS[k]} {((weightsNormalized?.[k] ?? 0) * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      </div>
      <div id="cards" className="cards">
        {items.map((it) => (
          <ListingCard
            key={it.listing_id}
            listing={it}
            componentKeys={componentKeys}
            onClick={onCardClick}
          />
        ))}
      </div>
    </>
  );
}
