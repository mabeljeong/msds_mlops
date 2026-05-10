import { useCallback, useEffect, useMemo, useState } from "react";
import { apiBaseLabel, getHealth, getListings, postRank } from "./api.js";
import { DEFAULT_COMPONENT_KEYS, TOP_N } from "./constants.js";
import { applyFilters } from "./filters.js";
import { useDebounced } from "./hooks.js";
import ListingMap from "./components/RentMap.jsx";
import ResultsList from "./components/ResultsList.jsx";
import Sidebar from "./components/Sidebar.jsx";
import TopBar from "./components/TopBar.jsx";

const INITIAL_WEIGHTS = { walk: 50, safety: 50, transit: 50 };
const INITIAL_BEDS = new Set(["2"]);
const INITIAL_BATHS = new Set();

export default function App() {
  // ----- Server state ----- //
  const [apiStatus, setApiStatus] = useState({ kind: "", text: "connecting…" });
  const [componentKeys, setComponentKeys] = useState(DEFAULT_COMPONENT_KEYS);
  const [allListings, setAllListings] = useState([]);
  const [ranked, setRanked] = useState([]);
  const [weightsNormalized, setWeightsNormalized] = useState({});
  const [rankStatus, setRankStatus] = useState("");
  const [focusListingId, setFocusListingId] = useState(null);

  // ----- Filter / weight state ----- //
  const [weights, setWeights] = useState(INITIAL_WEIGHTS);
  const [budget, setBudget] = useState("4000");
  const [bedsActive, setBedsActive] = useState(INITIAL_BEDS);
  const [bathsActive, setBathsActive] = useState(INITIAL_BATHS);
  const [neighborhoodNames, setNeighborhoodNames] = useState([]);

  const setWeight = useCallback(
    (key, value) => setWeights((prev) => ({ ...prev, [key]: value })),
    []
  );

  // Apartment-style toggle: clicking a pill flips its active state, capped
  // at two per group. When a third is added, the oldest is dropped so the
  // user always sees a clean exact-or-range selection.
  const togglePill = useCallback((group, value) => {
    const setter = group === "beds" ? setBedsActive : setBathsActive;
    setter((prev) => {
      const next = new Set(prev);
      if (next.has(value)) {
        next.delete(value);
        return next;
      }
      if (next.size >= 2) {
        const oldest = next.values().next().value;
        next.delete(oldest);
      }
      next.add(value);
      return next;
    });
  }, []);

  const clearPills = useCallback((group) => {
    const setter = group === "beds" ? setBedsActive : setBathsActive;
    setter(new Set());
  }, []);

  // ----- Initial load ----- //
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await getHealth();
        if (cancelled) return;
        if (Array.isArray(data.rank_component_keys) && data.rank_component_keys.length > 0) {
          setComponentKeys(data.rank_component_keys);
        }
        setApiStatus({
          kind: "ok",
          text: `model: ${data.model_source}${data.model_loaded ? " · loaded" : ""}`,
        });
      } catch {
        if (cancelled) return;
        setApiStatus({ kind: "err", text: `API unreachable @ ${apiBaseLabel()}` });
      }

      setRankStatus("loading listings…");
      try {
        const data = await getListings();
        if (cancelled) return;
        setAllListings(data.listings || []);
        setRankStatus(`${(data.listings || []).length} listings loaded`);
      } catch (err) {
        if (cancelled) return;
        setAllListings([]);
        setRankStatus("Failed to load listings");
        console.error(err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // ----- Re-rank whenever inputs change ----- //
  // Debounce the inputs together so dragging a slider or typing a budget
  // doesn't fire a request on every keystroke.
  const rankInputs = useMemo(
    () => ({
      weights,
      budget: budget === "" ? null : Number(budget),
      bedsActive,
      bathsActive,
      neighborhoodNames,
    }),
    [weights, budget, bedsActive, bathsActive, neighborhoodNames]
  );
  const debouncedInputs = useDebounced(rankInputs, 150);

  useEffect(() => {
    if (!allListings.length) return;
    let cancelled = false;
    (async () => {
      const filtered = applyFilters(allListings, debouncedInputs);
      if (!filtered.length) {
        setRanked([]);
        setRankStatus("No listings match the filters");
        setWeightsNormalized({});
        return;
      }
      setRankStatus("ranking…");
      try {
        const data = await postRank({
          listings: filtered,
          weights: debouncedInputs.weights,
          top_n: TOP_N,
        });
        if (cancelled) return;
        setRanked(data.results || []);
        setWeightsNormalized(data.weights_normalized || {});
        setRankStatus(
          data.n_returned < data.n_input
            ? `Top ${data.n_returned} of ${data.n_input} matches`
            : `Ranked ${data.n_returned} matches`
        );
      } catch (err) {
        if (cancelled) return;
        console.error(err);
        setRankStatus(String(err.message || err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [allListings, debouncedInputs]);

  const handleCardClick = useCallback((listingId) => {
    setFocusListingId(listingId);
    // Reset after a tick so clicking the same card again still re-focuses.
    setTimeout(() => setFocusListingId(null), 0);
  }, []);

  return (
    <>
      <TopBar apiStatus={apiStatus} />
      <main className="layout">
        <Sidebar
          weights={weights}
          setWeight={setWeight}
          budget={budget}
          setBudget={setBudget}
          bedsActive={bedsActive}
          bathsActive={bathsActive}
          togglePill={togglePill}
          clearPills={clearPills}
          neighborhoodNames={neighborhoodNames}
          setNeighborhoodNames={setNeighborhoodNames}
          rankStatus={rankStatus}
        />
        <section className="main">
          <ListingMap
            items={ranked}
            onMarkerClick={(id) => {
              const card = document.getElementById(`card-${id}`);
              if (card) {
                card.scrollIntoView({ behavior: "smooth", block: "center" });
                card.style.outline = "2px solid var(--accent)";
                setTimeout(() => (card.style.outline = ""), 1500);
              }
            }}
            focusListingId={focusListingId}
          />
          <ResultsList
            items={ranked}
            componentKeys={componentKeys}
            weightsNormalized={weightsNormalized}
            onCardClick={handleCardClick}
          />
        </section>
      </main>
    </>
  );
}
