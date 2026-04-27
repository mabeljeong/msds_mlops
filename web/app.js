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
  document.querySelectorAll(".slider-row").forEach((row) => {
    const range = row.querySelector("input[type=range]");
    const valueEl = row.querySelector(".value");
    valueEl.textContent = range.value;
    range.addEventListener("input", () => {
      valueEl.textContent = range.value;
    });
    range.addEventListener("change", rerank);
  });
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
  card.className = "listing-card" + (it.flag_overpriced ? " flagged" : "");
  card.id = `card-${it.listing_id}`;

  const title = it.title || it.address || `${it.bedrooms || 0}-bed in ${it.zip_code}`;
  const subtitle = it.address || `ZIP ${it.zip_code}`;

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

    ${rentBarHtml(it)}

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
      <span>${it.fair_rent_p10 != null && it.fair_rent_p90 != null
          ? `$${Math.round(it.fair_rent_p10).toLocaleString()} – $${Math.round(it.fair_rent_p90).toLocaleString()}`
          : "—"}</span>
    </div>

    <div class="components">
      ${COMPONENT_KEYS.map((k) => componentHtml(k, it.component_scores[k])).join("")}
    </div>

    <div class="card-foot">
      <span>${it.bedrooms || 0} bed${it.bathrooms ? ` · ${it.bathrooms} bath` : ""} · ZIP ${it.zip_code}</span>
      ${it.url ? `<a href="${it.url}" target="_blank" rel="noopener">view ↗</a>` : ""}
      ${it.flag_overpriced ? `<span class="flag-pill" title="${it.flag_reason || ""}">overpriced</span>` : ""}
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

function rentBarHtml(it) {
  const actual = Number(it.actual_rent_usd) || 0;
  const p10 = it.fair_rent_p10;
  const p90 = it.fair_rent_p90;
  const predicted = Number(it.predicted_rent_usd) || 0;

  const candidates = [actual, predicted];
  if (p10 != null) candidates.push(p10);
  if (p90 != null) candidates.push(p90);
  const lo = Math.min(...candidates) * 0.85;
  const hi = Math.max(...candidates) * 1.1;

  const pct = (v) => `${((v - lo) / (hi - lo)) * 100}%`;

  const bandHtml =
    p10 != null && p90 != null
      ? `<div class="rent-band" style="left:${pct(p10)}; width:${((p90 - p10) / (hi - lo)) * 100}%"></div>`
      : "";

  let markerCls = "rent-marker";
  if (p90 != null && actual > p90) markerCls += " over";
  else if (p10 != null && actual < p10) markerCls += " under";

  return `
    <div class="rent-bar" title="actual=$${Math.round(actual)} · predicted=$${Math.round(predicted)}">
      <div class="rent-axis"></div>
      ${bandHtml}
      <div class="rent-line" style="left:${pct(predicted)}"></div>
      <div class="${markerCls}" style="left:${pct(actual)}"></div>
      ${p10 != null ? `<div class="rent-tick" style="left:${pct(p10)}">$${Math.round(p10 / 100) / 10}k</div>` : ""}
      ${p90 != null ? `<div class="rent-tick" style="left:${pct(p90)}">$${Math.round(p90 / 100) / 10}k</div>` : ""}
      <div class="rent-tick predicted" style="left:${pct(actual)}; bottom: 32px;">
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
