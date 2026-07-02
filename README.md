# 🗺️ POI Polygon Extractor Chatbot

Interactive Streamlit chatbot: paste any real-estate / mall / POI link (or a plain address) → it extracts the **PARENT polygon** (lot / site / parcel) and **CHILD polygons** (building footprints), computes true geodesic areas, plots them on a satellite map, and exports to CSV/Excel (including appending to an existing Excel file).

## Features
- 💬 Chat interface — paste links, addresses, or ask questions about extracted data
- 🟦 Parent polygon (site/lot) + 🟥 child polygons (buildings) with m² / sqft / acres
- 🔁 Cross-reference fallback: if the listing page is blocked/unreadable, it recovers the address from the URL slug (LLM-assisted) and pulls polygons from OpenStreetMap instead
- 🗺️ Folium map with street + satellite layers
- ⬇️ Download CSV, new Excel, or **merge into an existing .xlsx**
- 📏 Compares listed lot size vs measured polygon area (% difference)
- Works even **without** an API key (regex fallback); Gemini key unlocks smarter extraction + chat

## Free APIs used
| Purpose | Service | Cost |
|---|---|---|
| LLM extraction + chat | Google Gemini (`gemini-2.0-flash`) | Free tier — get key at https://aistudio.google.com/apikey |
| Blocked-page reading | Jina AI Reader (`r.jina.ai`) | Free |
| Geocoding | OpenStreetMap Nominatim | Free |
| Polygons | Overpass API (OSM) | Free |
| Satellite tiles | Esri World Imagery | Free |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Enter your Gemini API key in the sidebar (optional but recommended).

## Deploy on Streamlit Community Cloud (free)
1. Push `app.py` + `requirements.txt` to a GitHub repo
2. Go to https://share.streamlit.io → New app → pick the repo → deploy
3. (Optional) add `GEMINI_API_KEY` via app Secrets and read it with `st.secrets`

## How parent vs child is decided
1. **Parent** = largest OSM `landuse`/`amenity`/`leisure` polygon containing the geocoded point → else Nominatim's own polygon for the address → else a square synthesized from the listed lot size (flagged as approximate).
2. **Children** = all OSM `building` footprints intersecting the parent (or within the search radius).
3. Areas are geodesic (WGS84 ellipsoid via `pyproj.Geod`) — accurate anywhere on Earth.

## Notes / limits
- Big portals (realestate.com.au, Zillow) aggressively block bots — that's exactly why the Jina reader + URL-slug cross-reference fallback exists.
- OSM has no legal cadastral parcels in most countries; the "parent" is the best available site polygon. For legal boundaries, plug in a cadastre API (e.g. NT ILIS / state land services) in `fetch_osm_polygons`.
- Respect Nominatim/Overpass usage policies (1 req/sec) for bulk runs.
