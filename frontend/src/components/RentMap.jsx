import { useEffect, useMemo, useRef } from "react";
import L from "leaflet";
import { MapContainer, Marker, Popup, TileLayer, useMap } from "react-leaflet";
import { neighborhoodLabelForZip } from "../constants.js";
import { priceStatus } from "../filters.js";

function fmtUsd(v) {
  return v == null ? "—" : `$${Math.round(v).toLocaleString()}`;
}

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function makeIcon(rank, kind) {
  return L.divIcon({
    html: `<div class="score-marker pin-${kind}">${rank}</div>`,
    className: "",
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });
}

function FitBounds({ items }) {
  const map = useMap();
  useEffect(() => {
    const bounds = items
      .filter((it) => it.lat != null && it.lng != null)
      .map((it) => [it.lat, it.lng]);
    if (bounds.length) {
      map.fitBounds(bounds, { padding: [30, 30], maxZoom: 13 });
    }
  }, [items, map]);
  return null;
}

export default function RentMap({ items, onMarkerClick, focusListingId }) {
  const markerRefs = useRef(new Map());

  const markers = useMemo(
    () =>
      items
        .filter((it) => it.lat != null && it.lng != null)
        .map((it) => ({ it, status: priceStatus(it) })),
    [items]
  );

  // When the user clicks a card, open the corresponding marker's popup
  // and pan the map to it.
  useEffect(() => {
    if (focusListingId == null) return;
    const ref = markerRefs.current.get(focusListingId);
    if (ref) {
      ref.openPopup();
    }
  }, [focusListingId]);

  return (
    <div className="map-wrap">
      <MapContainer
        center={[37.7749, -122.4194]}
        zoom={12}
        style={{ height: "420px" }}
        zoomControl
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> · &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          subdomains={["a", "b", "c", "d"]}
          maxZoom={19}
        />
        <FitBounds items={items} />
        {markers.map(({ it, status }) => {
          const neighborhood = neighborhoodLabelForZip(it.zip_code);
          const popupHtml = `
            <div style="min-width: 180px;">
              <div class="popup-title">#${it.rank} · ${escapeHtml(it.zip_code)}</div>
              ${
                neighborhood
                  ? `<div class="popup-sub">${escapeHtml(neighborhood)}</div>`
                  : ""
              }
              <div class="popup-meta">${it.bedrooms || 0} bed · ${fmtUsd(
                it.actual_rent_usd
              )}</div>
              <div class="popup-meta" style="margin-top:4px;">Fair band: ${fmtUsd(
                it.fair_rent_p25
              )} – ${fmtUsd(it.fair_rent_p75)}</div>
              <div class="popup-meta">Score: <strong>${(it.composite_score * 100).toFixed(0)}</strong>/100</div>
            </div>`;
          return (
            <Marker
              key={it.listing_id}
              position={[it.lat, it.lng]}
              icon={makeIcon(it.rank, status.kind)}
              eventHandlers={{
                click: () => onMarkerClick?.(it.listing_id),
              }}
              ref={(ref) => {
                if (ref) markerRefs.current.set(it.listing_id, ref);
                else markerRefs.current.delete(it.listing_id);
              }}
            >
              <Popup>
                <div dangerouslySetInnerHTML={{ __html: popupHtml }} />
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
      <div className="map-legend" aria-hidden="true">
        <span className="legend-item">
          <span className="legend-dot under" />
          Good deal
        </span>
        <span className="legend-item">
          <span className="legend-dot fair" />
          Fair price
        </span>
        <span className="legend-item">
          <span className="legend-dot over" />
          Overpriced
        </span>
      </div>
    </div>
  );
}
