// Number of ranked listings shown in "Top picks". The server filters with
// the full set of matches; we cap the visible cards/markers to this many.
export const TOP_N = 15;

// Fallback ordering until /health responds. Server is the source of truth.
export const DEFAULT_COMPONENT_KEYS = ["safety", "walk", "transit"];

export const COMPONENT_LABELS = {
  safety: "Safety",
  walk: "Walk",
  transit: "Transit",
};

// SF neighborhoods → 5-digit ZIPs covering each area. Listings only carry
// `zip_code`, so we filter by the union of ZIPs for the selected
// neighborhoods. A ZIP can appear under multiple neighborhoods (e.g. SoMa
// shares 94103 with Mission); the union dedupes naturally via Set.
export const SF_NEIGHBORHOODS = {
  "Bayview / Hunters Point": ["94124"],
  "Bernal Heights": ["94110"],
  "Castro / Noe Valley": ["94114", "94131"],
  Chinatown: ["94108", "94133"],
  "Excelsior / Outer Mission": ["94112"],
  "Financial District / Embarcadero": ["94104", "94105", "94111"],
  "Glen Park / Diamond Heights": ["94131"],
  "Haight-Ashbury": ["94117"],
  "Hayes Valley / Civic Center": ["94102"],
  "Lake Merced / Parkmerced": ["94132"],
  "Marina / Cow Hollow": ["94123"],
  Mission: ["94110", "94103"],
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
  Tenderloin: ["94102"],
  "Treasure Island": ["94130"],
  "Twin Peaks / West Portal": ["94127"],
};

/** ZIP -> sorted unique neighborhood names. Built once. */
export const ZIP_TO_NEIGHBORHOODS = (() => {
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
    ])
  );
})();

export function neighborhoodLabelForZip(zip) {
  if (zip == null) return "";
  const names = ZIP_TO_NEIGHBORHOODS.get(String(zip));
  return names && names.length ? names.join(" · ") : "";
}

/**
 * Bed/bath pill scales. Each entry has:
 *   value: stable id (also used as React key + dataset)
 *   label: visible text
 *   n:     numeric edge — exact integer for beds, lower bound for baths
 *   unbounded: true when this pill represents "and up"
 */
export const BED_PILLS = [
  { value: "0", label: "Studio", n: 0, unbounded: false },
  { value: "1", label: "1", n: 1, unbounded: false },
  { value: "2", label: "2", n: 2, unbounded: false },
  { value: "3", label: "3", n: 3, unbounded: false },
  { value: "4+", label: "4+", n: 4, unbounded: true },
];

export const BATH_PILLS = [
  { value: "1", label: "1+", n: 1, unbounded: false },
  { value: "1.5", label: "1.5+", n: 1.5, unbounded: false },
  { value: "2", label: "2+", n: 2, unbounded: false },
  { value: "3+", label: "3+", n: 3, unbounded: true },
];
