# RentRadar Frontend (React + Vite)

The user-facing UI for the RentRadar API. Talks to the FastAPI backend in
[`../app/`](../app/).

The legacy vanilla SPA in [`../web/`](../web/) is preserved as a fallback
during the React migration; FastAPI prefers `frontend/dist/` when it exists
and falls back to `web/` otherwise.

## Tech

- **React 18** + **Vite 5** for the build / dev server
- **react-leaflet 4** + **leaflet 1.9** for the map
- No CSS framework — plain CSS variables in [`src/styles.css`](src/styles.css)
  (ported from the original `web/styles.css`)

## Local development

You need **Node 20+** and **npm 10+**.

```bash
# Once, to install deps
cd frontend
npm install
```

Then run the React dev server and the FastAPI server in **two terminals**:

```bash
# Terminal 1 — FastAPI on http://localhost:8000
cd ..
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — React dev server on http://localhost:5173
cd frontend
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/health`, `/listings`, `/rank`,
`/predict`, `/flag_overpriced` to the FastAPI server (see
[`vite.config.js`](vite.config.js)), so same-origin `fetch()` calls Just
Work even though the two servers are on different ports.

> Override the API target with `VITE_API_TARGET=http://other-host:8000 npm run dev`
> if you need to point the dev proxy somewhere else.

## Production build

```bash
cd frontend
npm run build      # outputs dist/
npm run preview    # serves dist/ on http://localhost:4173 for sanity checks
```

Once `dist/` exists, you can run the FastAPI server alone and it will serve
the built React app at <http://localhost:8000> (no Node needed at runtime):

```bash
cd ..
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

This is exactly what the production Docker image does — see
[`../Dockerfile`](../Dockerfile).

## Project layout

```
frontend/
├── index.html              # Vite entry HTML; loads /src/main.jsx
├── package.json
├── vite.config.js          # Dev server proxy + build options
└── src/
    ├── main.jsx            # React root
    ├── App.jsx             # Top-level state + data flow
    ├── api.js              # fetch wrappers around the FastAPI endpoints
    ├── constants.js        # SF neighborhood ZIPs, bed/bath pill scales
    ├── filters.js          # Pure helpers: filtering + price status logic
    ├── hooks.js            # useDebounced
    ├── styles.css
    └── components/
        ├── TopBar.jsx
        ├── Sidebar.jsx
        ├── RentMap.jsx
        ├── ResultsList.jsx
        ├── ListingCard.jsx
        └── RentBar.jsx
```
