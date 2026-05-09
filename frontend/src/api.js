// API base resolution:
//   - In dev (`npm run dev`), Vite proxies /health, /listings, /rank, etc.
//     to the FastAPI server (see vite.config.js). Use empty string so fetches
//     are same-origin and pass through the proxy.
//   - In production (Docker / Cloud Run), FastAPI serves the built frontend
//     and exposes the API on the same host. Same-origin works there too.
//   - Override with VITE_API_BASE at build time if the API ever lives on a
//     different host.
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function jsonOrThrow(res) {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return jsonOrThrow(res);
}

export async function getListings() {
  const res = await fetch(`${API_BASE}/listings`);
  return jsonOrThrow(res);
}

export async function postRank({ listings, weights, top_n }) {
  const res = await fetch(`${API_BASE}/rank`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ listings, weights, top_n }),
  });
  return jsonOrThrow(res);
}

export const apiBaseLabel = () => API_BASE || window.location.origin;
