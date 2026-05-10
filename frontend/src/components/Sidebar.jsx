import { BATH_PILLS, BED_PILLS, SF_NEIGHBORHOODS } from "../constants.js";

const NEIGHBORHOOD_NAMES = Object.keys(SF_NEIGHBORHOODS).sort((a, b) => a.localeCompare(b));

function PillRow({ label, scale, active, onToggle, onClear, hint }) {
  return (
    <div className="filter-row pill-row">
      <div className="pill-row-head">
        <span className="pill-label">{label}</span>
        <button
          type="button"
          className="pill-clear"
          onClick={onClear}
          aria-label={`Clear ${label.toLowerCase()} filter`}
        >
          Any
        </button>
      </div>
      <div className="pill-group" role="group" aria-label={label}>
        {scale.map((spec) => {
          const pressed = active.has(spec.value);
          return (
            <button
              key={spec.value}
              type="button"
              className="pill"
              aria-pressed={pressed}
              onClick={() => onToggle(spec.value)}
            >
              {spec.label}
            </button>
          );
        })}
      </div>
      <p className="muted pill-hint">{hint}</p>
    </div>
  );
}

export default function Sidebar({
  weights,
  setWeight,
  budget,
  setBudget,
  bedsActive,
  togglePill,
  clearPills,
  bathsActive,
  neighborhoodNames,
  setNeighborhoodNames,
  rankStatus,
}) {
  return (
    <aside className="sidebar">
      <section className="card">
        <h2>Your priorities</h2>
        <p className="muted">Drag to weight what matters. Scores auto-normalize.</p>

        {[
          { key: "walk", label: "Walkability" },
          { key: "safety", label: "Safety" },
          { key: "transit", label: "Transit" },
        ].map(({ key, label }) => (
          <div className="slider-row" key={key}>
            <label>
              {label} <span className="value">{weights[key]}</span>
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={weights[key]}
              onChange={(e) => setWeight(key, Number(e.target.value))}
            />
          </div>
        ))}
      </section>

      <section className="card">
        <h2>Filters</h2>
        <label className="filter-row">
          Budget (USD/mo)
          <input
            type="number"
            min="500"
            step="100"
            placeholder="e.g. 3500"
            value={budget}
            onChange={(e) => setBudget(e.target.value)}
          />
        </label>

        <PillRow
          label="Bedrooms"
          scale={BED_PILLS}
          active={bedsActive}
          onToggle={(v) => togglePill("beds", v)}
          onClear={() => clearPills("beds")}
          hint="Tap one for exact, or two for a range."
        />

        <PillRow
          label="Bathrooms"
          scale={BATH_PILLS}
          active={bathsActive}
          onToggle={(v) => togglePill("baths", v)}
          onClear={() => clearPills("baths")}
          hint="Each pill is a minimum; pick two to bound a range."
        />

        <label className="filter-row neighborhood-row">
          Neighborhoods
          <select
            multiple
            size="6"
            value={neighborhoodNames}
            onChange={(e) =>
              setNeighborhoodNames(
                Array.from(e.target.selectedOptions, (o) => o.value)
              )
            }
          >
            {NEIGHBORHOOD_NAMES.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <p className="muted neighborhood-hint">
          ⌘/Ctrl-click to pick multiple. Empty = all of SF.
        </p>

        <div className="muted" id="rankStatus">
          {rankStatus}
        </div>
      </section>

      <section className="card legend">
        <h2>How a card reads</h2>
        <p className="muted">
          The bar shows the model's <strong>p25–p75 fair-rent band</strong>.
          The <span className="dot dot-actual" /> is the actual asking rent.
          Inside the band is fair; right of the band means we flag it as{" "}
          <em>overpriced</em>.
        </p>
      </section>
    </aside>
  );
}
