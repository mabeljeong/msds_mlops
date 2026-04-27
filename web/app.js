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

const COMPONENT_KEYS = ["price_fairness", "safety", "walk", "transit", "affordability"];
const COMPONENT_LABELS = {
  price_fairness: "Fair",
  affordability: "Afford",
  safety: "Safety",
  walk: "Walk",
  transit: "Transit",
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
  initMap();
  await pollHealth();
  await loadListings();
  await rerank();

  document.getElementById("rankBtn").addEventListener("click", rerank);
  document.getElementById("budget").addEventListener("change", rerank);
  document.getElementById("beds").addEventListener("change", rerank);
  document.getElementById("zipFilter").addEventListener("change", rerank);
});

// ----- API health --------------------------------------------------------- //
async function pollHealth() {
  const el = document.getElementById("apiStatus");
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
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

function applyFilters(listings) {
  const beds = document.getElementById("beds").value;
  const zip = document.getElementById("zipFilter").value.trim();
  return listings.filter((l) => {
    if (beds !== "" && Math.floor(Number(l.bedrooms || 0)) !== Number(beds)) {
      // Special case: bedrooms field can be 0 for studio
      if (!(beds === "3" && Number(l.bedrooms || 0) >= 3)) return false;
    }
    if (zip && String(l.zip_code) !== zip) return false;
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
  const budgetVal = document.getElementById("budget").value;
  const body = {
    listings: filtered,
    weights,
    budget_usd: budgetVal ? Number(budgetVal) : null,
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
    status.textContent = `Ranked ${data.n_returned}/${data.n_input}`;

    const w = data.weights_normalized || {};
    const wStr = COMPONENT_KEYS.map((k) => `${COMPONENT_LABELS[k]} ${(w[k] * 100).toFixed(0)}%`).join(" · ");
    document.getElementById("resultsMeta").textContent = wStr;

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
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);
  state.map = map;
}

function colorForScore(score) {
  // 0 → red, 0.5 → yellow, 1 → green
  const s = Math.max(0, Math.min(1, score));
  const r = s < 0.5 ? 239 : Math.round(239 + (34 - 239) * ((s - 0.5) / 0.5));
  const g = s < 0.5 ? Math.round(68 + (204 - 68) * (s / 0.5)) : Math.round(204 + (197 - 204) * ((s - 0.5) / 0.5));
  const b = s < 0.5 ? Math.round(68 + (21 - 68) * (s / 0.5)) : Math.round(21 + (94 - 21) * ((s - 0.5) / 0.5));
  return `rgb(${r}, ${g}, ${b})`;
}

function renderMap(items) {
  const map = state.map;
  if (!map) return;

  state.markers.forEach((m) => map.removeLayer(m));
  state.markers.clear();

  const bounds = [];
  items.forEach((it) => {
    if (it.lat == null || it.lng == null) return;
    const color = colorForScore(it.composite_score);
    const html = `<div class="score-marker" style="background:${color}">${it.rank}</div>`;
    const icon = L.divIcon({
      html,
      className: "",
      iconSize: [24, 24],
      iconAnchor: [12, 12],
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
  return `
    <div style="font-family: inherit; min-width: 180px;">
      <div style="font-weight:600;">#${it.rank} · ${it.zip_code}</div>
      <div style="color:#475569; font-size: 12px;">${it.bedrooms || 0} bed · ${fmt(it.actual_rent_usd)}</div>
      <div style="margin-top:4px;">Fair band: ${fmt(it.fair_rent_p10)} – ${fmt(it.fair_rent_p90)}</div>
      <div>Score: <strong>${(it.composite_score * 100).toFixed(0)}</strong>/100</div>
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
    it.fair_rent_p10 != null && it.fair_rent_p90 != null
      ? `$${Math.round(it.fair_rent_p10).toLocaleString()} – $${Math.round(it.fair_rent_p90).toLocaleString()}`
      : `~$${Math.round(it.predicted_rent_usd * 0.9).toLocaleString()} – $${Math.round(it.predicted_rent_usd * 1.1).toLocaleString()} (est.)`;

  card.innerHTML = `
    <div class="card-head">
      <div class="title-block">
        <p class="title" title="${escapeHtml(title)}">${escapeHtml(title)}</p>
        <p class="subtitle" title="${escapeHtml(subtitle)}">${escapeHtml(subtitle)}</p>
      </div>
      <div class="rank">#${it.rank}</div>
    </div>

    <div class="kv-row">
      <span class="label">Composite</span>
      <span class="score-badge" style="background: ${badgeBg(it.composite_score)}">
        <span class="num">${(it.composite_score * 100).toFixed(0)}</span>/100
      </span>
    </div>

    <div class="kv-row">
      <span class="label">Price status</span>
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
      <span class="label">Fair band p10–p90</span>
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
 * Decide a price-status verdict (`fair` / `over` / `under`) for the actual rent
 * relative to the fair-rent band returned by /rank. We prefer the calibrated p10/p90
 * band; if it's unavailable (e.g. placeholder model) we fall back to predicted ±10%.
 * `flag_overpriced` from the backend is the source of truth and overrides the visual
 * verdict when set.
 */
/**
 * Per-listing price verdict. Pure function of this listing's `actual_rent_usd`
 * vs its own fair-rent band. Nothing is hard-coded across listings.
 *
 *   actual > p90  →  "over"   (right of band)
 *   actual < p10  →  "under"  (left of band)
 *   otherwise     →  "fair"   (inside band)
 *
 * If the backend returned a calibrated band (real MLflow model), p10/p90 are used
 * directly and the label is "Overpriced" / "Fair" / "Under fair". If only the
 * placeholder is loaded (no band), we synthesize predicted ± 10% and append "(est.)"
 * to the label so the user knows the comparison is uncalibrated — but the verdict
 * still varies per listing based on its own price.
 */
function priceStatus(it) {
  const actual = Number(it.actual_rent_usd) || 0;
  const predicted = Number(it.predicted_rent_usd) || 0;
  const hasBand = it.fair_rent_p10 != null && it.fair_rent_p90 != null;
  const p10 = hasBand ? it.fair_rent_p10 : predicted * 0.9;
  const p90 = hasBand ? it.fair_rent_p90 : predicted * 1.1;

  let kind = "fair";
  if (it.flag_overpriced || actual > p90) kind = "over";
  else if (actual < p10) kind = "under";

  const baseLabel = kind === "over" ? "Overpriced" : kind === "under" ? "Under fair" : "Fair";
  const label = hasBand ? baseLabel : `${baseLabel} (est.)`;
  const tooltip = hasBand
    ? `Actual $${Math.round(actual)} vs fair band $${Math.round(p10)}–$${Math.round(p90)}`
    : `Actual $${Math.round(actual)} vs estimated band $${Math.round(p10)}–$${Math.round(p90)} (placeholder model — set MLFLOW_MODEL_URI for a calibrated band).`;

  return { kind, label, tooltip, p10, p90, hasBand };
}

function rentBarHtml(it, status) {
  const actual = Number(it.actual_rent_usd) || 0;
  const predicted = Number(it.predicted_rent_usd) || 0;
  const { p10, p90, hasBand } = status;

  // Bar spans both the band and the actual rent, with padding, so the marker is
  // always inside the visible bar regardless of whether the listing is over/under.
  const lo = Math.min(actual, p10) * 0.9;
  const hi = Math.max(actual, p90) * 1.1;

  const clamp = (n) => Math.max(2, Math.min(98, n));
  const pct = (v) => `${clamp(((v - lo) / (hi - lo)) * 100)}%`;

  const bandWidthPct = clamp(((p90 - p10) / (hi - lo)) * 100);
  const bandClass = hasBand ? "rent-band" : "rent-band synthetic";
  const bandTitle = hasBand
    ? `Fair band p10–p90: $${Math.round(p10)}–$${Math.round(p90)}`
    : `Estimated band (predicted ±10%): $${Math.round(p10)}–$${Math.round(p90)}`;

  const predictedLine = hasBand
    ? `<div class="rent-line" style="left:${pct(predicted)}" title="Predicted: $${Math.round(predicted)}"></div>`
    : "";

  const markerCls = `rent-marker ${status.kind}`;

  return `
    <div class="rent-bar" title="Actual $${Math.round(actual)} · Predicted $${Math.round(predicted)}${
      hasBand ? "" : " (placeholder)"
    }">
      <div class="rent-axis"></div>
      <div class="${bandClass}" style="left:${pct(p10)}; width:${bandWidthPct}%" title="${escapeHtml(bandTitle)}"></div>
      ${predictedLine}
      <div class="${markerCls}" style="left:${pct(actual)}"></div>
      <div class="rent-tick" style="left:${pct(p10)}">$${(p10 / 1000).toFixed(1)}k</div>
      <div class="rent-tick" style="left:${pct(p90)}">$${(p90 / 1000).toFixed(1)}k</div>
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

function badgeBg(score) {
  const s = Math.max(0, Math.min(1, score));
  // light fill matching colorForScore
  const c = colorForScore(s);
  return c.replace("rgb", "rgba").replace(")", ", 0.18)");
}

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
