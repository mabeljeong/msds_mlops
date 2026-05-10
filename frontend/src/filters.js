import { BATH_PILLS, BED_PILLS, SF_NEIGHBORHOODS } from "./constants.js";

/**
 * Sort active pill values for a group by position on the scale,
 * returning the matching pill spec objects (lowest first).
 */
function sortedActivePills(scale, active) {
  const byValue = new Map(scale.map((p, i) => [p.value, { spec: p, idx: i }]));
  return Array.from(active)
    .map((v) => byValue.get(v))
    .filter(Boolean)
    .sort((a, b) => a.idx - b.idx)
    .map(({ spec }) => spec);
}

/**
 * Apartment-style pill matcher:
 *   0 active → no constraint
 *   1 active → "exact bucket" for that pill (bedrooms: integer match or ≥
 *              for "+"; bathrooms: ≥ pill threshold)
 *   2 active → inclusive range [low.n, high.n]; if the high pill is
 *              unbounded (e.g. 4+ beds, 3+ baths), the upper cap is removed.
 */
function matchesPillSelection(scale, active, raw, { floorValue, singleMinThreshold = false }) {
  if (!active || active.size === 0) return true;
  if (raw == null || raw === "" || Number.isNaN(Number(raw))) return false;
  const n = floorValue ? Math.floor(Number(raw)) : Number(raw);
  const pills = sortedActivePills(scale, active);
  if (pills.length === 0) return true;
  if (pills.length === 1) {
    const p = pills[0];
    if (singleMinThreshold) return n >= p.n;
    return p.unbounded ? n >= p.n : n === p.n;
  }
  const [lo, hi] = [pills[0], pills[pills.length - 1]];
  if (n < lo.n) return false;
  if (hi.unbounded) return true;
  return n <= hi.n;
}

export function matchesBedPillSelection(active, raw) {
  return matchesPillSelection(BED_PILLS, active, raw, { floorValue: true });
}

/**
 * Bath filters use minimum semantics (1+ means ≥ 1). Demo listings often
 * omit `bathrooms`; treat missing as 1.0 so 1+ still surfaces results.
 */
export function matchesBathPillSelection(active, raw) {
  let effective = raw;
  if (raw == null || raw === "" || Number.isNaN(Number(raw))) {
    effective = 1;
  }
  return matchesPillSelection(BATH_PILLS, active, effective, {
    floorValue: false,
    singleMinThreshold: true,
  });
}

export function selectedNeighborhoodZips(selectedNames) {
  const zips = new Set();
  for (const name of selectedNames) {
    for (const zip of SF_NEIGHBORHOODS[name] || []) zips.add(zip);
  }
  return zips;
}

export function applyFilters(listings, { bedsActive, bathsActive, budget, neighborhoodNames }) {
  const zipAllow = selectedNeighborhoodZips(neighborhoodNames);
  return listings.filter((l) => {
    if (!matchesBedPillSelection(bedsActive, l.bedrooms)) return false;
    if (bathsActive.size > 0 && !matchesBathPillSelection(bathsActive, l.bathrooms)) return false;
    if (budget != null && budget > 0) {
      const rent = Number(l.actual_rent_usd ?? l.rent_usd);
      if (!Number.isFinite(rent) || rent > budget) return false;
    }
    if (zipAllow.size > 0 && !zipAllow.has(String(l.zip_code))) return false;
    return true;
  });
}

/**
 * Per-listing price verdict. Pure function of this listing's
 * `actual_rent_usd` vs its own fair-rent band.
 *
 *   actual > p75  →  "over"
 *   actual < p25  →  "under"
 *   otherwise     →  "fair"
 *
 * If the backend returned a calibrated band, p25/p75 are used directly. If
 * only the placeholder is loaded (no band), we synthesize predicted ± 10%
 * and append "(est.)" so the user knows the comparison is uncalibrated.
 */
export function priceStatus(it) {
  const actual = Number(it.actual_rent_usd) || 0;
  const predicted = Number(it.predicted_rent_usd) || 0;
  const hasBand = it.fair_rent_p25 != null && it.fair_rent_p75 != null;
  const p25 = hasBand ? it.fair_rent_p25 : predicted * 0.9;
  const p75 = hasBand ? it.fair_rent_p75 : predicted * 1.1;

  let kind = "fair";
  if (it.flag_overpriced || actual > p75) kind = "over";
  else if (actual < p25) kind = "under";

  const baseLabel = kind === "over" ? "Overpriced" : kind === "under" ? "Good deal" : "Fair price";
  const label = hasBand ? baseLabel : `${baseLabel} (est.)`;
  const tooltip = hasBand
    ? `Actual $${Math.round(actual)} vs fair band $${Math.round(p25)}–$${Math.round(p75)}`
    : `Actual $${Math.round(actual)} vs estimated band $${Math.round(p25)}–$${Math.round(p75)} (placeholder model — set MLFLOW_MODEL_URI for a calibrated band).`;

  return { kind, label, tooltip, p25, p75, hasBand };
}
