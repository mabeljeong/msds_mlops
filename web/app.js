/* RentIQ — single-page client.
 *
 * Talks to the FastAPI service at the same origin (or http://localhost:8000 if served
 * from disk via file://). Pulls listings from /listings, then re-ranks via /rank
 * whenever weights or filters change.
 */

const API_BASE = (() => {
  if (location.protocol === "file:" || !location.host) return "http://localhost:8000";
  return ""; // same-origin
})();

// Number of ranked listings shown in "Top picks". The server filters with the
// full set of matches; we cap the visible cards/markers to this many.
const TOP_N = 15;

// Fallback ordering used until /health responds (or if it omits the field for
// any reason). The server is the source of truth — see pollHealth() below.
let COMPONENT_KEYS = ["safety", "walk", "transit"];
const COMPONENT_LABELS = {
  safety: "Safety",
  walk: "Walk",
  transit: "Transit",
};

// SF neighborhoods → 5-digit ZIPs covering each area. Listings only carry
// `zip_code`, so we filter by the union of ZIPs for the selected neighborhoods.
// A ZIP can appear under multiple neighborhoods (e.g. SoMa shares 94103 with
// Mission); the union dedupes naturally via Set.
const SF_NEIGHBORHOODS = {
  "Bayview / Hunters Point": ["94124"],
  "Bernal Heights": ["94110"],
  "Castro / Noe Valley": ["94114", "94131"],
  "Chinatown": ["94108", "94133"],
  "Excelsior / Outer Mission": ["94112"],
  "Financial District / Embarcadero": ["94104", "94105", "94111"],
  "Glen Park / Diamond Heights": ["94131"],
  "Haight-Ashbury": ["94117"],
  "Hayes Valley / Civic Center": ["94102"],
  "Lake Merced / Parkmerced": ["94132"],
  "Marina / Cow Hollow": ["94123"],
  "Mission": ["94110", "94103"],
  "Nob Hill / Russian Hill": ["94109"],
  "North Beach / Telegraph Hill": ["94133"],
  "Pacific Heights / Western Addition": ["94115"],
  "Portola / Visitacion Valley": ["94134"],
  "Potrero Hill / Dogpatch": ["94107"],
  "Presidio / Sea Cliff": ["94129"],
  "Inner Richmond": ["94118"],
  "Outer Richmond": ["94121"],
  "Inner Sunset": ["94122"],
  "Outer Sunset / Parkside": ["94116"],
  "SoMa / Mission Bay": ["94103", "94107", "94158"],
  "Tenderloin": ["94102"],
  "Treasure Island": ["94130"],
  "Twin Peaks / West Portal": ["94127"],
};

/** ZIP -> sorted unique neighborhood names. Built once from SF_NEIGHBORHOODS. */
const ZIP_TO_NEIGHBORHOODS = (() => {
  const out = new Map();
  for (const [name, zips] of Object.entries(SF_NEIGHBORHOODS)) {
    for (const zip of zips) {
      const key = String(zip);
      if (!out.has(key)) out.set(key, new Set());
      out.get(key).add(name);
    }
  }
  return new Map(
    Array.from(out.entries()).map(([zip, names]) => [
      zip,
      Array.from(names).sort((a, b) => a.localeCompare(b)),
    ]),
  );
})();

function neighborhoodLabelForZip(zip) {
  if (zip == null) return "";
  const names = ZIP_TO_NEIGHBORHOODS.get(String(zip));
  return names && names.length ? names.join(" · ") : "";
}

/**
 * Bed/bath pill scales. Each entry has:
 *   value: stable id for the pill (also used in dataset)
 *   label: visible text
 *   n:     numeric edge — exact integer for beds, lower bound for baths
 *   unbounded: true when this pill represents "and up" (no upper cap when used as range max)
 */
const BED_PILLS = [
  { value: "0", label: "Studio", n: 0, unbounded: false },
  { value: "1", label: "1", n: 1, unbounded: false },
  { value: "2", label: "2", n: 2, unbounded: false },
  { value: "3", label: "3", n: 3, unbounded: false },
  { value: "4+", label: "4+", n: 4, unbounded: true },
];

const BATH_PILLS = [
  { value: "1", label: "1+", n: 1, unbounded: false },
  { value: "1.5", label: "1.5+", n: 1.5, unbounded: false },
  { value: "2", label: "2+", n: 2, unbounded: false },
  { value: "3+", label: "3+", n: 3, unbounded: true },
];

/** Active pill values per group. Order does not matter — we sort by scale index. */
const filterState = {
  beds: new Set(["2"]),
  baths: new Set(),
};

/** Live state */
const state = {
  listings: [],         // raw listings from /listings
  ranked: [],           // RankedListing[] from /rank
  map: null,
  markers: new Map(),   // listing_id -> Leaflet marker
};

// ----- Boot --------------------------------------------------------------- //
document.addEventListener("DOMContentLoaded", async () => {
  initSliders();
  initRoomPills();
  initNeighborhoods();
  initMap();
  await pollHealth();
  await loadListings();
  await rerank();

  document.getElementById("rankBtn").addEventListener("click", rerank);
  document.getElementById("budget").addEventListener("change", rerank);
  document.getElementById("neighborhoodFilter").addEventListener("change", rerank);
});

// ----- API health --------------------------------------------------------- //
async function pollHealth() {
  const el = document.getElementById("apiStatus");
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    if (Array.isArray(data.rank_component_keys) && data.rank_component_keys.length > 0) {
      COMPONENT_KEYS = data.rank_component_keys;
    }
    el.textContent = `model: ${data.model_source}${data.model_loaded ? " · loaded" : ""}`;
    el.classList.add("ok");
  } catch (err) {
    el.textContent = `API unreachable @ ${API_BASE || location.origin}`;
    el.classList.add("err");
  }
}

// ----- Listings + ranking ------------------------------------------------- //
async function loadListings() {
  const status = document.getElementById("rankStatus");
  status.textContent = "loading listings…";
  try {
    const res = await fetch(`${API_BASE}/listings`);
    const data = await res.json();
    state.listings = data.listings || [];
    status.textContent = `${state.listings.length} listings loaded`;
  } catch (err) {
    state.listings = [];
    status.textContent = "Failed to load listings";
    console.error(err);
  }
}

function readWeights() {
  const w = {};
  document.querySelectorAll(".slider-row").forEach((row) => {
    const key = row.dataset.key;
    const v = Number(row.querySelector("input[type=range]").value);
    w[key] = v;
  });
  return w;
}

/**
 * Sort the active pill values for a group by their position on the scale,
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
 *   1 active → "exact bucket" for that pill (bedrooms: integer match or ≥ for "+";
 *              bathrooms: ≥ pill threshold)
 *   2 active → inclusive range [low.n, high.n]; if the high pill is unbounded
 *              (e.g. 4+ beds, 3+ baths), the upper cap is removed.
 *
 * Beds use floor() so 1.0/1.5 both count as "1". Baths use the raw number so
 * 1.5+ behaves correctly.
 *
 * `singleMinThreshold`: when true (bathrooms), a single active pill means
 * listing >= threshold ("1+", "2+", …), not exact equality.
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

function matchesBedPillSelection(raw) {
  return matchesPillSelection(BED_PILLS, filterState.beds, raw, { floorValue: true });
}

/**
 * Bath filters use minimum semantics (1+ means >= 1). Demo listings often omit
 * `bathrooms`; treat missing as 1.0 so 1+ still surfaces results (stricter
 * tiers like 1.5+ correctly require known counts or imputed 1 failing 1.5+).
 */
function matchesBathPillSelection(raw) {
  let effective = raw;
  if (raw == null || raw === "" || Number.isNaN(Number(raw))) {
    effective = 1;
  }
  return matchesPillSelection(BATH_PILLS, filterState.baths, effective, {
    floorValue: false,
    singleMinThreshold: true,
  });
}

function selectedNeighborhoodZips() {
  const select = document.getElementById("neighborhoodFilter");
  const zips = new Set();
  for (const opt of select.selectedOptions) {
    for (const zip of SF_NEIGHBORHOODS[opt.value] || []) zips.add(zip);
  }
  return zips;
}

function applyFilters(listings) {
  const budgetRaw = document.getElementById("budget").value;
  const budget = budgetRaw ? Number(budgetRaw) : null;
  const zipAllow = selectedNeighborhoodZips();

  return listings.filter((l) => {
    if (!matchesBedPillSelection(l.bedrooms)) return false;
    if (filterState.baths.size > 0 && !matchesBathPillSelection(l.bathrooms)) return false;
    if (budget != null && budget > 0) {
      const rent = Number(l.actual_rent_usd ?? l.rent_usd);
      if (!Number.isFinite(rent) || rent > budget) return false;
    }
    if (zipAllow.size > 0 && !zipAllow.has(String(l.zip_code))) return false;
    return true;
  });
}

async function rerank() {
  const status = document.getElementById("rankStatus");
  if (!state.listings.length) {
    status.textContent = "No listings loaded";
    return;
  }

  const filtered = applyFilters(state.listings);
  if (!filtered.length) {
    state.ranked = [];
    renderCards([]);
    renderMap([]);
    status.textContent = "No listings match the filters";
    document.getElementById("resultsMeta").textContent = "";
    return;
  }

  const weights = readWeights();
  const body = {
    listings: filtered,
    weights,
    top_n: TOP_N,
  };

  status.textContent = "ranking…";
  try {
    const res = await fetch(`${API_BASE}/rank`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`rank failed (${res.status}): ${err.slice(0, 200)}`);
    }
    const data = await res.json();
    state.ranked = data.results || [];
    status.textContent =
      data.n_returned < data.n_input
        ? `Top ${data.n_returned} of ${data.n_input} matches`
        : `Ranked ${data.n_returned} matches`;

    const w = data.weights_normalized || {};
    const meta = document.getElementById("resultsMeta");
    meta.innerHTML = COMPONENT_KEYS.map(
      (k) =>
        `<span class="weight-chip">${escapeHtml(COMPONENT_LABELS[k])} ${(w[k] * 100).toFixed(0)}%</span>`,
    ).join("");

    renderCards(state.ranked);
    renderMap(state.ranked);
  } catch (err) {
    console.error(err);
    status.textContent = String(err.message || err);
  }
}

// ----- Sliders ------------------------------------------------------------ //
function initSliders() {
  const debouncedRerank = debounce(rerank, 150);
  document.querySelectorAll(".slider-row").forEach((row) => {
    const range = row.querySelector("input[type=range]");
    const valueEl = row.querySelector(".value");
    valueEl.textContent = range.value;
    range.addEventListener("input", () => {
      valueEl.textContent = range.value;
      debouncedRerank();
    });
  });
}

// ----- Bed/bath pills ----------------------------------------------------- //
function initRoomPills() {
  const debouncedRerank = debounce(rerank, 100);

  const groups = [
    { id: "bedsPills", scale: BED_PILLS, key: "beds" },
    { id: "bathsPills", scale: BATH_PILLS, key: "baths" },
  ];

  for (const { id, scale, key } of groups) {
    const container = document.getElementById(id);
    if (!container) continue;
    container.innerHTML = "";
    for (const spec of scale) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pill";
      btn.textContent = spec.label;
      btn.dataset.value = spec.value;
      btn.setAttribute("aria-pressed", filterState[key].has(spec.value) ? "true" : "false");
      btn.addEventListener("click", () => {
        togglePill(key, spec.value);
        renderPillGroup(key);
        debouncedRerank();
      });
      container.appendChild(btn);
    }
  }

  document.querySelectorAll(".pill-clear").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.clear;
      if (!key) return;
      filterState[key].clear();
      renderPillGroup(key);
      debouncedRerank();
    });
  });
}

/**
 * Apartment-style toggle: clicking a pill flips its active state. We cap each
 * group at two active pills (drop the oldest when a third is added) so the user
 * always sees a clean exact-or-range selection.
 */
function togglePill(key, value) {
  const set = filterState[key];
  if (set.has(value)) {
    set.delete(value);
    return;
  }
  if (set.size >= 2) {
    const oldest = set.values().next().value;
    set.delete(oldest);
  }
  set.add(value);
}

function renderPillGroup(key) {
  const containerId = key === "beds" ? "bedsPills" : "bathsPills";
  const container = document.getElementById(containerId);
  if (!container) return;
  const set = filterState[key];
  for (const btn of container.querySelectorAll(".pill")) {
    btn.setAttribute("aria-pressed", set.has(btn.dataset.value) ? "true" : "false");
  }
}

function initNeighborhoods() {
  const select = document.getElementById("neighborhoodFilter");
  if (!select) return;
  const names = Object.keys(SF_NEIGHBORHOODS).sort((a, b) => a.localeCompare(b));
  for (const name of names) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    select.appendChild(opt);
  }
}

function debounce(fn, wait) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), wait);
  };
}

// ----- Map ---------------------------------------------------------------- //
function initMap() {
  const map = L.map("map", { zoomControl: true }).setView([37.7749, -122.4194], 12);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 19,
    subdomains: "abcd",
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> · &copy; <a href="https://carto.com/attributions">CARTO</a>',
  }).addTo(map);
  state.map = map;
}

function renderMap(items) {
  const map = state.map;
  if (!map) return;

  state.markers.forEach((m) => map.removeLayer(m));
  state.markers.clear();

  const bounds = [];
  items.forEach((it) => {
    if (it.lat == null || it.lng == null) return;
    const status = priceStatus(it);
    const html = `<div class="score-marker pin-${status.kind}">${it.rank}</div>`;
    const icon = L.divIcon({
      html,
      className: "",
      iconSize: [28, 28],
      iconAnchor: [14, 14],
    });
    const marker = L.marker([it.lat, it.lng], { icon }).addTo(map);
    marker.bindPopup(popupHtml(it));
    marker.on("click", () => {
      const card = document.getElementById(`card-${it.listing_id}`);
      if (card) {
        card.scrollIntoView({ behavior: "smooth", block: "center" });
        card.style.outline = "2px solid var(--accent)";
        setTimeout(() => (card.style.outline = ""), 1500);
      }
    });
    state.markers.set(it.listing_id, marker);
    bounds.push([it.lat, it.lng]);
  });

  if (bounds.length) {
    map.fitBounds(bounds, { padding: [30, 30], maxZoom: 13 });
  }
}

function popupHtml(it) {
  const fmt = (v) => (v == null ? "—" : `$${Math.round(v).toLocaleString()}`);
  const neighborhood = neighborhoodLabelForZip(it.zip_code);
  const neighborhoodLine = neighborhood
    ? `<div class="popup-sub">${escapeHtml(neighborhood)}</div>`
    : "";
  return `
    <div style="min-width: 180px;">
      <div class="popup-title">#${it.rank} · ${it.zip_code}</div>
      ${neighborhoodLine}
      <div class="popup-meta">${it.bedrooms || 0} bed · ${fmt(it.actual_rent_usd)}</div>
      <div class="popup-meta" style="margin-top:4px;">Fair band: ${fmt(it.fair_rent_p25)} – ${fmt(it.fair_rent_p75)}</div>
      <div class="popup-meta">Score: <strong>${(it.composite_score * 100).toFixed(0)}</strong>/100</div>
    </div>`;
}

// ----- Cards -------------------------------------------------------------- //
function renderCards(items) {
  const container = document.getElementById("cards");
  container.innerHTML = "";
  document.getElementById("resultsTitle").textContent =
    items.length ? `Top picks (${items.length})` : "No matches";

  for (const it of items) {
    container.appendChild(buildCard(it));
  }
}

function buildCard(it) {
  const card = document.createElement("article");
  const status = priceStatus(it);
  card.className = `listing-card status-${status.kind}` + (it.flag_overpriced ? " flagged" : "");
  card.id = `card-${it.listing_id}`;

  const title = it.title || it.address || `${it.bedrooms || 0}-bed in ${it.zip_code}`;
  const subtitle = it.address || `ZIP ${it.zip_code}`;

  const bandLabel =
    it.fair_rent_p25 != null && it.fair_rent_p75 != null
      ? `$${Math.round(it.fair_rent_p25).toLocaleString()} – $${Math.round(it.fair_rent_p75).toLocaleString()}`
      : `~$${Math.round(it.predicted_rent_usd * 0.9).toLocaleString()} – $${Math.round(it.predicted_rent_usd * 1.1).toLocaleString()} (est.)`;

  const neighborhood = neighborhoodLabelForZip(it.zip_code);
  const neighborhoodHtml = neighborhood
    ? `<p class="neighborhood" title="${escapeHtml(neighborhood)}">${escapeHtml(neighborhood)}</p>`
    : "";

  card.innerHTML = `
    <div class="card-head">
      <div class="title-block">
        ${neighborhoodHtml}
        <p class="title" title="${escapeHtml(title)}">${escapeHtml(title)}</p>
        <p class="subtitle" title="${escapeHtml(subtitle)}">${escapeHtml(subtitle)}</p>
      </div>
      <div class="rank">#${it.rank}</div>
    </div>

    <div class="composite-row">
      <div class="composite-block">
        <span class="score-label">Composite score</span>
        <span class="score-badge">
          <span class="num">${(it.composite_score * 100).toFixed(0)}</span><span class="denom">/100</span>
        </span>
      </div>
      <span class="price-pill ${status.kind}" title="${escapeHtml(status.tooltip)}">${status.label}</span>
    </div>

    ${rentBarHtml(it, status)}

    <div class="kv-row">
      <span class="label">Asking</span>
      <span>$${Math.round(it.actual_rent_usd).toLocaleString()}/mo</span>
    </div>
    <div class="kv-row">
      <span class="label">Predicted (fair)</span>
      <span>$${Math.round(it.predicted_rent_usd).toLocaleString()}/mo</span>
    </div>
    <div class="kv-row">
      <span class="label">Fair band p25–p75</span>
      <span>${bandLabel}</span>
    </div>

    <div class="components">
      ${COMPONENT_KEYS.map((k) => componentHtml(k, it.component_scores[k])).join("")}
    </div>

    <div class="card-foot">
      <span>${it.bedrooms || 0} bed${it.bathrooms ? ` · ${it.bathrooms} bath` : ""} · ZIP ${it.zip_code}</span>
      ${it.url ? `<a href="${it.url}" target="_blank" rel="noopener">view ↗</a>` : ""}
      ${it.flag_overpriced ? `<span class="flag-pill" title="${escapeHtml(it.flag_reason || "")}">overpriced</span>` : ""}
    </div>
  `;

  card.addEventListener("click", () => {
    const m = state.markers.get(it.listing_id);
    if (m && state.map) {
      state.map.setView(m.getLatLng(), 14, { animate: true });
      m.openPopup();
    }
  });

  return card;
}

/**
 * Per-listing price verdict. Pure function of this listing's `actual_rent_usd`
 * vs its own fair-rent band. Nothing is hard-coded across listings.
 *
 *   actual > p75  →  "over"   (right of band)
 *   actual < p25  →  "under"  (left of band)
 *   otherwise     →  "fair"   (inside band)
 *
 * If the backend returned a calibrated band (real MLflow model), p25/p75 are used
 * directly and the label is "Overpriced" / "Fair" / "Under fair". If only the
 * placeholder is loaded (no band), we synthesize predicted ± 10% and append "(est.)"
 * to the label so the user knows the comparison is uncalibrated — but the verdict
 * still varies per listing based on its own price.
 */
function priceStatus(it) {
  const actual = Number(it.actual_rent_usd) || 0;
  const predicted = Number(it.predicted_rent_usd) || 0;
  const hasBand = it.fair_rent_p25 != null && it.fair_rent_p75 != null;
  const p25 = hasBand ? it.fair_rent_p25 : predicted * 0.9;
  const p75 = hasBand ? it.fair_rent_p75 : predicted * 1.1;

  let kind = "fair";
  if (it.flag_overpriced || actual > p75) kind = "over";
  else if (actual < p25) kind = "under";

  const baseLabel = kind === "over" ? "Overpriced" : kind === "under" ? "Under fair" : "Fair";
  const label = hasBand ? baseLabel : `${baseLabel} (est.)`;
  const tooltip = hasBand
    ? `Actual $${Math.round(actual)} vs fair band $${Math.round(p25)}–$${Math.round(p75)}`
    : `Actual $${Math.round(actual)} vs estimated band $${Math.round(p25)}–$${Math.round(p75)} (placeholder model — set MLFLOW_MODEL_URI for a calibrated band).`;

  return { kind, label, tooltip, p25, p75, hasBand };
}

function rentBarHtml(it, status) {
  const actual = Number(it.actual_rent_usd) || 0;
  const predicted = Number(it.predicted_rent_usd) || 0;
  const { p25, p75, hasBand } = status;

  // Bar spans both the band and the actual rent, with padding, so the marker is
  // always inside the visible bar regardless of whether the listing is over/under.
  const lo = Math.min(actual, p25) * 0.9;
  const hi = Math.max(actual, p75) * 1.1;

  const clamp = (n) => Math.max(2, Math.min(98, n));
  const pct = (v) => `${clamp(((v - lo) / (hi - lo)) * 100)}%`;

  const bandWidthPct = clamp(((p75 - p25) / (hi - lo)) * 100);
  const bandClass = hasBand ? "rent-band" : "rent-band synthetic";
  const bandTitle = hasBand
    ? `Fair band p25–p75: $${Math.round(p25)}–$${Math.round(p75)}`
    : `Estimated band (predicted ±10%): $${Math.round(p25)}–$${Math.round(p75)}`;

  const predictedLine = hasBand
    ? `<div class="rent-line" style="left:${pct(predicted)}" title="Predicted: $${Math.round(predicted)}"></div>`
    : "";

  const markerCls = `rent-marker ${status.kind}`;

  return `
    <div class="rent-bar" title="Actual $${Math.round(actual)} · Predicted $${Math.round(predicted)}${
      hasBand ? "" : " (placeholder)"
    }">
      <div class="rent-axis"></div>
      <div class="${bandClass}" style="left:${pct(p25)}; width:${bandWidthPct}%" title="${escapeHtml(bandTitle)}"></div>
      ${predictedLine}
      <div class="${markerCls}" style="left:${pct(actual)}"></div>
      <div class="rent-tick" style="left:${pct(p25)}">$${(p25 / 1000).toFixed(1)}k</div>
      <div class="rent-tick" style="left:${pct(p75)}">$${(p75 / 1000).toFixed(1)}k</div>
      <div class="rent-tick actual-tick" style="left:${pct(actual)}; bottom: 32px;">
        $${Math.round(actual).toLocaleString()}
      </div>
    </div>
  `;
}

function componentHtml(key, score) {
  const pct = Math.max(0, Math.min(1, Number(score) || 0));
  return `
    <div class="comp" title="${COMPONENT_LABELS[key]}: ${pct.toFixed(2)}">
      ${COMPONENT_LABELS[key]}
      <div class="bar"><div style="width:${(pct * 100).toFixed(0)}%"></div></div>
      <span class="pct">${(pct * 100).toFixed(0)}</span>
    </div>
  `;
}

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
