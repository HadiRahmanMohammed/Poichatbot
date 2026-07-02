"""
POI Polygon Extractor Chatbot — v1.0
=====================================
Interactive Streamlit chatbot that extracts PARENT (lot / site / parcel) and
CHILD (building footprint) polygons for any property / mall / POI from a
real-estate or general web link.

Pipeline:
  1. Scrape the shared link (direct fetch -> Jina AI reader fallback, both free)
  2. LLM (Google Gemini, free tier) extracts address + listed lot/building size
  3. Geocode via OpenStreetMap Nominatim (free)
  4. Fetch polygons via Nominatim polygon_geojson + Overpass API (free)
     - PARENT  = enclosing land parcel / landuse / site polygon
     - CHILD   = building footprints inside the parent
  5. Geodesic area computed with pyproj (m² + sqft + acres)
  6. Results -> table, CSV/Excel download, append to an existing Excel file
  7. Folium map with parent (blue) and child (red) polygons

Run:  streamlit run app.py
"""

import io
import json
import math
import re
import time
from datetime import datetime

import folium
import pandas as pd
import requests
import streamlit as st
from pyproj import Geod
from shapely.geometry import Point, Polygon, shape
from streamlit_folium import st_folium

# ----------------------------------------------------------------------------
# Constants & helpers
# ----------------------------------------------------------------------------
GEOD = Geod(ellps="WGS84")
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
}
NOMINATIM = "https://nominatim.openstreetmap.org"
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

M2_TO_SQFT = 10.7639
M2_TO_ACRE = 0.000247105

PARENT_TAGS = [
    "landuse", "amenity", "leisure", "shop", "tourism", "boundary", "place",
]


def geodesic_area_m2(geom) -> float:
    """Geodesic (true earth-surface) area of a shapely polygon, in m²."""
    try:
        area, _ = GEOD.geometry_area_perimeter(geom)
        return abs(area)
    except Exception:
        return 0.0


def fmt_area(m2: float) -> str:
    return f"{m2:,.1f} m² | {m2 * M2_TO_SQFT:,.0f} sqft | {m2 * M2_TO_ACRE:.3f} ac"


# ----------------------------------------------------------------------------
# 1) Page fetching (direct -> Jina reader fallback)
# ----------------------------------------------------------------------------
def fetch_page_text(url: str) -> tuple[str, str]:
    """Return (text, method). Tries direct fetch, then r.jina.ai reader."""
    # Direct
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if r.status_code == 200 and len(r.text) > 500:
            txt = re.sub(r"<script.*?</script>", " ", r.text, flags=re.S | re.I)
            txt = re.sub(r"<style.*?</style>", " ", txt, flags=re.S | re.I)
            txt = re.sub(r"<[^>]+>", " ", txt)
            txt = re.sub(r"\s+", " ", txt)
            if len(txt) > 400:
                return txt[:18000], "direct"
    except Exception:
        pass
    # Jina AI reader (free, renders JS, bypasses many blocks)
    try:
        r = requests.get(f"https://r.jina.ai/{url}", headers=UA, timeout=25)
        if r.status_code == 200 and len(r.text) > 200:
            return r.text[:18000], "jina-reader"
    except Exception:
        pass
    return "", "failed"


# ----------------------------------------------------------------------------
# 2) LLM extraction (Gemini) + regex fallback
# ----------------------------------------------------------------------------
def gemini_call(prompt: str, api_key: str, model: str, temperature: float = 0.1) -> str:
    if not api_key:
        return ""
    try:
        resp = requests.post(
            GEMINI_URL.format(model=model),
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": 1024},
            },
            timeout=40,
        )
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""


def regex_extract(text: str) -> dict:
    """Cheap fallback extraction when no LLM key is set."""
    out = {"address": None, "lot_size_m2": None, "building_size_m2": None}
    # land / lot size
    m = re.search(r"(?:land|lot)\s*(?:size|area)?[:\s]*([\d,\.]+)\s*(m2|m²|sqm|sq\.?\s*m|acres?|ha|sq\.?\s*ft|sqft)", text, re.I)
    if m:
        val = float(m.group(1).replace(",", ""))
        unit = m.group(2).lower()
        if "ac" in unit:
            val /= M2_TO_ACRE
        elif unit == "ha":
            val *= 10000
        elif "ft" in unit:
            val /= M2_TO_SQFT
        out["lot_size_m2"] = val
    # address-ish line
    m = re.search(r"\d{1,5}[A-Za-z]?\s+[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,3}\s+(?:St|Street|Rd|Road|Ave|Avenue|Dr|Drive|Ct|Court|Cres|Crescent|Blvd|Boulevard|Ln|Lane|Way|Pl|Place|Tce|Terrace)\b[^,.\n]{0,40}(?:,\s*[^,.\n]{2,40}){0,3}", text)
    if m:
        out["address"] = m.group(0).strip()
    return out


def llm_extract_listing(text: str, url: str, api_key: str, model: str) -> dict:
    prompt = f"""You are a real-estate data extractor. From the page text below (source: {url}),
extract and return ONLY a JSON object (no markdown, no backticks) with keys:
  "address": full street address as a single string (or null),
  "property_type": e.g. house, apartment, mall, land, commercial (or null),
  "lot_size_m2": land/lot size converted to square metres as a number (or null),
  "building_size_m2": building/floor size in square metres as a number (or null),
  "listed_price": string or null,
  "notes": one short sentence of anything relevant to land area.
Convert acres (x4046.86), hectares (x10000), sqft (/10.7639) to m².

PAGE TEXT:
{text[:12000]}"""
    raw = gemini_call(prompt, api_key, model)
    if raw:
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
    return regex_extract(text)


# ----------------------------------------------------------------------------
# 3) Geocoding (Nominatim, returns polygon when available)
# ----------------------------------------------------------------------------
def geocode(query: str) -> dict | None:
    try:
        r = requests.get(
            f"{NOMINATIM}/search",
            params={
                "q": query, "format": "json", "limit": 1,
                "polygon_geojson": 1, "addressdetails": 1,
            },
            headers=UA, timeout=20,
        )
        res = r.json()
        if res:
            return res[0]
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------
# 4) Overpass polygons
# ----------------------------------------------------------------------------
def overpass(query: str) -> dict:
    for ep in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(ep, data={"data": query}, headers=UA, timeout=45)
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
    return {"elements": []}


def way_to_polygon(el, nodes: dict) -> Polygon | None:
    if el.get("type") != "way":
        return None
    coords = []
    if "geometry" in el:
        coords = [(g["lon"], g["lat"]) for g in el["geometry"]]
    else:
        coords = [(nodes[n][1], nodes[n][0]) for n in el.get("nodes", []) if n in nodes]
    if len(coords) >= 4 and coords[0] == coords[-1]:
        try:
            p = Polygon(coords)
            if p.is_valid and p.area > 0:
                return p
        except Exception:
            pass
    return None


def fetch_osm_polygons(lat: float, lon: float, radius: int = 120) -> dict:
    """Returns {'parent': [(polygon, tags)], 'children': [(polygon, tags)]}."""
    q = f"""
    [out:json][timeout:40];
    (
      way(around:{radius},{lat},{lon})["building"];
      way(around:{radius},{lat},{lon})["landuse"];
      way(around:{radius},{lat},{lon})["amenity"];
      way(around:{radius},{lat},{lon})["leisure"];
      way(around:{radius},{lat},{lon})["shop"]["building"!~".*"];
      relation(around:{radius},{lat},{lon})["building"];
    );
    out geom tags;
    """
    data = overpass(q)
    pt = Point(lon, lat)
    parents, children = [], []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        poly = None
        if el["type"] == "way":
            poly = way_to_polygon(el, {})
        elif el["type"] == "relation" and "members" in el:
            outers = [m for m in el["members"] if m.get("role") == "outer" and "geometry" in m]
            if outers:
                try:
                    coords = [(g["lon"], g["lat"]) for g in outers[0]["geometry"]]
                    if len(coords) >= 4:
                        poly = Polygon(coords)
                except Exception:
                    poly = None
        if poly is None or not poly.is_valid:
            continue
        if "building" in tags:
            children.append((poly, tags))
        elif any(k in tags for k in PARENT_TAGS):
            if poly.contains(pt) or poly.distance(pt) < 0.0005:
                parents.append((poly, tags))
    return {"parents": parents, "children": children}


# ----------------------------------------------------------------------------
# 5) Full pipeline for one URL / address
# ----------------------------------------------------------------------------
def run_pipeline(query: str, api_key: str, model: str, radius: int, progress) -> dict:
    result = {
        "source": query, "address": None, "lat": None, "lon": None,
        "listing": {}, "rows": [], "geojson_parent": [], "geojson_children": [],
        "log": [], "fallback_used": False,
    }
    is_url = query.lower().startswith("http")
    listing = {}

    if is_url:
        progress.write("🌐 Fetching page…")
        text, method = fetch_page_text(query)
        result["log"].append(f"Page fetch: {method}")
        if text:
            progress.write(f"🤖 Extracting listing data ({'Gemini' if api_key else 'regex'})…")
            listing = llm_extract_listing(text, query, api_key, model)
        else:
            result["log"].append("Direct link unreadable — falling back to cross-reference.")
        # Cross-reference fallback: derive a search query from the URL slug
        if not listing.get("address"):
            slug = re.sub(r"https?://[^/]+/", "", query)
            slug = re.sub(r"[-_/]+", " ", slug)
            slug = re.sub(r"\b(property|listing|details|for sale|html?|www|com|au)\b", " ", slug, flags=re.I)
            slug = re.sub(r"\d{6,}", " ", slug)
            slug = re.sub(r"\s+", " ", slug).strip()
            if api_key and slug:
                guess = gemini_call(
                    f"This URL slug came from a real-estate listing: '{slug}'. "
                    f"Return ONLY the most likely full street address (one line, no extra words).",
                    api_key, model,
                )
                if guess and len(guess) < 120:
                    listing["address"] = guess.strip()
                    result["fallback_used"] = True
                    result["log"].append("Address recovered from URL slug via LLM (cross-reference mode).")
            elif slug:
                listing["address"] = slug
                result["fallback_used"] = True
    else:
        listing = {"address": query}

    result["listing"] = listing
    addr = listing.get("address")
    if not addr:
        result["log"].append("❌ Could not determine an address from this link.")
        return result

    progress.write(f"📍 Geocoding: {addr}")
    geo = geocode(addr)
    if not geo:
        geo = geocode(re.sub(r"^[^,]+,\s*", "", addr))  # retry without unit/street no.
    if not geo:
        result["log"].append("❌ Geocoding failed.")
        return result

    lat, lon = float(geo["lat"]), float(geo["lon"])
    result.update(lat=lat, lon=lon, address=geo.get("display_name", addr))

    # Parent candidate #1: Nominatim's own polygon for the address
    nominatim_parent = None
    gj = geo.get("geojson")
    if gj and gj.get("type") in ("Polygon", "MultiPolygon"):
        try:
            g = shape(gj)
            if g.geom_type == "MultiPolygon":
                g = max(g.geoms, key=lambda p: p.area)
            nominatim_parent = g
        except Exception:
            pass

    progress.write("🗺️ Querying OpenStreetMap polygons (Overpass)…")
    osm = fetch_osm_polygons(lat, lon, radius)

    # Choose parent: prefer landuse/site polygon containing point, else Nominatim polygon
    parent_poly, parent_tags = None, {}
    if osm["parents"]:
        parent_poly, parent_tags = max(osm["parents"], key=lambda pt_: geodesic_area_m2(pt_[0]))
    if parent_poly is None and nominatim_parent is not None:
        parent_poly, parent_tags = nominatim_parent, {"source": "nominatim"}
    if parent_poly is None and listing.get("lot_size_m2"):
        # Synthesize a square parent from the listed lot size (approximation)
        side = math.sqrt(float(listing["lot_size_m2"]))
        d = side / 2 / 111320
        parent_poly = Polygon([
            (lon - d, lat - d), (lon + d, lat - d),
            (lon + d, lat + d), (lon - d, lat + d), (lon - d, lat - d),
        ])
        parent_tags = {"source": "listed lot size (synthesized square)"}
        result["fallback_used"] = True

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    if parent_poly is not None:
        a = geodesic_area_m2(parent_poly)
        result["geojson_parent"].append(json.loads(json.dumps(parent_poly.__geo_interface__)))
        result["rows"].append({
            "timestamp": ts, "polygon_type": "PARENT",
            "name": parent_tags.get("name", parent_tags.get("landuse", parent_tags.get("source", "site/lot"))),
            "area_m2": round(a, 1), "area_sqft": round(a * M2_TO_SQFT, 0),
            "area_acres": round(a * M2_TO_ACRE, 4),
            "address": result["address"], "lat": lat, "lon": lon,
            "osm_tags": json.dumps(parent_tags)[:200], "source_url": query,
            "listed_lot_m2": listing.get("lot_size_m2"),
        })

    # Children: buildings intersecting the parent (or within radius if no parent)
    kept = 0
    for poly, tags in osm["children"]:
        if parent_poly is not None and not poly.intersects(parent_poly.buffer(0.0002)):
            continue
        a = geodesic_area_m2(poly)
        if a < 4:
            continue
        result["geojson_children"].append(json.loads(json.dumps(poly.__geo_interface__)))
        result["rows"].append({
            "timestamp": ts, "polygon_type": "CHILD",
            "name": tags.get("name", tags.get("building", "building")),
            "area_m2": round(a, 1), "area_sqft": round(a * M2_TO_SQFT, 0),
            "area_acres": round(a * M2_TO_ACRE, 4),
            "address": result["address"], "lat": lat, "lon": lon,
            "osm_tags": json.dumps(tags)[:200], "source_url": query,
            "listed_lot_m2": None,
        })
        kept += 1

    result["log"].append(f"✅ Parent: {'found' if parent_poly is not None else 'NOT found'} | Children: {kept}")
    return result


# ----------------------------------------------------------------------------
# 6) Map builder
# ----------------------------------------------------------------------------
def build_map(res: dict) -> folium.Map:
    m = folium.Map(location=[res["lat"], res["lon"]], zoom_start=18, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Street").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite",
    ).add_to(m)
    for gj in res["geojson_parent"]:
        folium.GeoJson(
            gj, name="Parent polygon",
            style_function=lambda x: {"color": "#1f6feb", "weight": 3, "fillColor": "#1f6feb", "fillOpacity": 0.10},
            tooltip="PARENT (lot / site)",
        ).add_to(m)
    for gj in res["geojson_children"]:
        folium.GeoJson(
            gj, name="Child polygon",
            style_function=lambda x: {"color": "#d1242f", "weight": 2, "fillColor": "#d1242f", "fillOpacity": 0.30},
            tooltip="CHILD (building)",
        ).add_to(m)
    folium.Marker([res["lat"], res["lon"]], tooltip=res["address"]).add_to(m)
    folium.LayerControl().add_to(m)
    return m


# ----------------------------------------------------------------------------
# 7) Excel helpers
# ----------------------------------------------------------------------------
def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="POI_Polygons")
    return buf.getvalue()


def append_to_existing_excel(upload, df_new: pd.DataFrame) -> bytes:
    try:
        old = pd.read_excel(upload)
        combined = pd.concat([old, df_new], ignore_index=True)
    except Exception:
        combined = df_new
    return df_to_excel_bytes(combined)


# ----------------------------------------------------------------------------
# 8) Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="POI Polygon Extractor", page_icon="🗺️", layout="wide")

st.markdown(
    """
    <div style="padding:14px 18px;border-radius:12px;
         background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);color:white;">
      <h2 style="margin:0;">🗺️ POI Polygon Extractor <span style="font-size:0.55em;opacity:.7;">expert mode</span></h2>
      <p style="margin:4px 0 0;opacity:.85;">Paste a real-estate / mall / POI link → get parent & child polygon areas, map, and Excel export.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API key (free — aistudio.google.com)", type="password")
    model = st.selectbox("Gemini model", ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"])
    radius = st.slider("Search radius around POI (m)", 40, 500, 120, 20)
    st.caption("No key? The app still works — it falls back to regex extraction + OSM. "
               "The key improves address extraction from messy listing pages and enables the chat assistant.")
    st.divider()
    st.subheader("📊 Excel")
    existing_xlsx = st.file_uploader("Append results to existing Excel", type=["xlsx"])
    if st.button("🧹 Clear session data"):
        st.session_state.clear()
        st.rerun()

# ---- Session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "G'day! 👋 Paste a property / mall / POI link (realestate.com.au, domain, Zillow, Google Maps place, etc.) "
                   "or just type an address, and I'll extract the **parent** (lot/site) and **child** (building) polygons with areas. "
                   "You can also ask me questions about the extracted data.",
    }]
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "last_result" not in st.session_state:
    st.session_state.last_result = None

URL_RE = re.compile(r"https?://\S+")
ADDR_HINT = re.compile(r"\d+\s+\w+.*(street|st\b|road|rd\b|ave|drive|dr\b|court|ct\b|cres|blvd|lane|ln\b|way|place|pl\b|terrace|tce)", re.I)

# ---- Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Chat input
user_input = st.chat_input("Paste a link or address, or ask about the data…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    urls = URL_RE.findall(user_input)
    looks_like_address = bool(ADDR_HINT.search(user_input))

    with st.chat_message("assistant"):
        if urls or looks_like_address:
            query = urls[0] if urls else user_input.strip()
            with st.status("Extracting polygons…", expanded=True) as progress:
                res = run_pipeline(query, api_key, model, radius, progress)
                progress.update(label="Done", state="complete", expanded=False)

            if res["rows"]:
                df_new = pd.DataFrame(res["rows"])
                st.session_state.df = pd.concat([st.session_state.df, df_new], ignore_index=True)
                st.session_state.last_result = res

                n_child = sum(1 for r in res["rows"] if r["polygon_type"] == "CHILD")
                parent_row = next((r for r in res["rows"] if r["polygon_type"] == "PARENT"), None)
                reply = f"**Extracted {len(res['rows'])} polygon(s)** for *{res['address']}*\n\n"
                if parent_row:
                    reply += f"- 🟦 **Parent** ({parent_row['name']}): {fmt_area(parent_row['area_m2'])}\n"
                    if parent_row.get("listed_lot_m2"):
                        diff = 100 * (parent_row["area_m2"] - parent_row["listed_lot_m2"]) / parent_row["listed_lot_m2"]
                        reply += f"  - Listing states {parent_row['listed_lot_m2']:,.0f} m² ({diff:+.1f}% vs polygon)\n"
                reply += f"- 🟥 **Children**: {n_child} building footprint(s)\n"
                if res["fallback_used"]:
                    reply += "\n> ℹ️ The original link couldn't be fully read, so I cross-referenced "
                    reply += "(URL slug → address → OpenStreetMap) to recover the data.\n"
                reply += "\nSee the **table, map and downloads** below 👇"
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            else:
                reply = ("I couldn't extract polygons for that one. 😕\n\n" +
                         "\n".join(f"- {l}" for l in res["log"]) +
                         "\n\n**Tips:** try pasting the property address directly, or a Google Maps link to the place.")
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            # Conversational turn — answer with Gemini using data context
            ctx = st.session_state.df.to_csv(index=False)[:6000] if not st.session_state.df.empty else "No data extracted yet."
            answer = gemini_call(
                f"You are a friendly GIS/POI assistant inside a Streamlit app. Extracted polygon data (CSV):\n{ctx}\n\n"
                f"User question: {user_input}\nAnswer concisely and helpfully.",
                api_key, model, temperature=0.6,
            ) or ("Add a Gemini API key in the sidebar to enable chat answers — "
                  "or paste a property link/address and I'll extract its polygons.")
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# ----------------------------------------------------------------------------
# Results section
# ----------------------------------------------------------------------------
if not st.session_state.df.empty:
    st.divider()
    tab_map, tab_table, tab_export = st.tabs(["🗺️ Map", "📋 Data table", "⬇️ Export"])

    with tab_map:
        if st.session_state.last_result and st.session_state.last_result.get("lat"):
            st_folium(build_map(st.session_state.last_result), width=None, height=520)
            st.caption("🟦 Parent = lot / site polygon 🟥 Red = child building footprints. Toggle satellite in the layer control.")

    with tab_table:
        st.dataframe(
            st.session_state.df,
            use_container_width=True,
            column_config={
                "area_m2": st.column_config.NumberColumn("Area (m²)", format="%.1f"),
                "area_sqft": st.column_config.NumberColumn("Area (sqft)", format="%.0f"),
                "area_acres": st.column_config.NumberColumn("Area (acres)", format="%.4f"),
            },
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Total polygons", len(st.session_state.df))
        c2.metric("Parents", int((st.session_state.df.polygon_type == "PARENT").sum()))
        c3.metric("Children", int((st.session_state.df.polygon_type == "CHILD").sum()))

    with tab_export:
        col1, col2, col3 = st.columns(3)
        col1.download_button(
            "⬇️ Download CSV",
            st.session_state.df.to_csv(index=False).encode(),
            file_name="poi_polygons.csv", mime="text/csv", use_container_width=True,
        )
        col2.download_button(
            "⬇️ Download Excel (new file)",
            df_to_excel_bytes(st.session_state.df),
            file_name="poi_polygons.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        if existing_xlsx is not None:
            col3.download_button(
                "⬇️ Download merged Excel (appended)",
                append_to_existing_excel(existing_xlsx, st.session_state.df),
                file_name=f"merged_{existing_xlsx.name}",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            col3.info("Upload an existing .xlsx in the sidebar to append & merge.")

st.markdown(
    "<p style='text-align:center;opacity:.5;font-size:.8em;margin-top:24px;'>"
    "Data © OpenStreetMap contributors · Geocoding: Nominatim · Polygons: Overpass API · LLM: Google Gemini (free tier)</p>",
    unsafe_allow_html=True,
)
