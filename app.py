"""
POI Polygon Extractor Chatbot — fixed deployable Streamlit version
-----------------------------------------------------------------
Paste a property / mall / POI URL, Google Maps URL, lat/lon, or address.
The app extracts a parent polygon and child building polygons using free services:
- Jina Reader for page text fallback
- Gemini API optional for smarter address extraction/chat
- Nominatim + Photon for geocoding
- Overpass/OpenStreetMap for polygons
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import time
from datetime import datetime
from urllib.parse import unquote, urlparse

import folium
import pandas as pd
import requests
import streamlit as st
from pyproj import Geod
from shapely.geometry import Point, Polygon, shape
from shapely.ops import unary_union
from streamlit_folium import st_folium

GEOD = Geod(ellps="WGS84")
M2_TO_SQFT = 10.7639
M2_TO_ACRE = 0.000247105
NOMINATIM = "https://nominatim.openstreetmap.org"
PHOTON = "https://photon.komoot.io/api/"
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
APP_EMAIL = os.getenv("CONTACT_EMAIL", "poi-extractor@example.com")
UA = {"User-Agent": f"POI Polygon Extractor/1.1 ({APP_EMAIL})", "Accept-Language": "en"}

STREET_WORDS = r"street|st|road|rd|avenue|ave|drive|dr|court|ct|crescent|cres|boulevard|blvd|lane|ln|way|place|pl|terrace|tce|highway|hwy|parade|pde|circuit|cct|esplanade|esp|circle|cir|trail|trl|parkway|pkwy|square|sq"
AU_STATES = "NSW VIC QLD WA SA TAS ACT NT".split()
US_STATES = "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split()
PARENT_KEYS = ["landuse", "amenity", "leisure", "shop", "tourism", "boundary", "place", "office"]


def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default) or os.getenv(name, default)
    except Exception:
        return os.getenv(name, default)


def geodesic_area_m2(geom) -> float:
    try:
        area, _ = GEOD.geometry_area_perimeter(geom)
        return abs(area)
    except Exception:
        return 0.0


def geom_to_wkt(geom) -> str:
    try:
        return geom.wkt
    except Exception:
        return ""


def fmt_area(m2: float) -> str:
    return f"{m2:,.1f} m² | {m2 * M2_TO_SQFT:,.0f} sqft | {m2 * M2_TO_ACRE:.3f} ac"


def clean_text(html: str) -> str:
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", html).strip()


def extract_latlon_from_url_or_text(text: str) -> tuple[float, float] | None:
    s = unquote(text)
    patterns = [
        r"@(-?\d+\.\d+),\s*(-?\d+\.\d+)",
        r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)",
        r"[?&](?:q|ll|center)=(-?\d+\.\d+),\s*(-?\d+\.\d+)",
        r"\b(-?\d{1,2}\.\d{4,})\s*,\s*(-?\d{1,3}\.\d{4,})\b",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
    return None


def tidy_slug(s: str) -> str:
    s = unquote(s).replace("+", " ").replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip(" ,/|")
    return s


def address_from_url(url: str) -> str | None:
    """Best-effort portal URL address extraction. Some portals only expose suburb+listing id."""
    u = urlparse(url.lower())
    host, path = u.netloc, unquote(u.path)

    if "zillow." in host:
        m = re.search(r"/homedetails/([^/]+)/", path)
        if m:
            return tidy_slug(re.sub(r"/.*", "", m.group(1))) + ", USA"

    if "redfin." in host:
        m = re.search(r"/([a-z]{2})/([^/]+)/([^/]+)/home/", path)
        if m:
            street = re.sub(r"\s\d{5}$", "", tidy_slug(m.group(3)))
            return f"{street}, {tidy_slug(m.group(2))}, {m.group(1).upper()}, USA"

    if "realtor.com" in host:
        m = re.search(r"/realestateandhomes-detail/([^/]+)", path)
        if m:
            return tidy_slug(re.sub(r"_m\d.*$", "", m.group(1)).replace("_", ", ")) + ", USA"

    if "domain.com.au" in host:
        m = re.search(rf"/(\d+[a-z]?(?:-[\w']+)+?-(?:{STREET_WORDS})-[\w'-]+-(?:{'|'.join(s.lower() for s in AU_STATES)})-\d{{4}})", path)
        if m:
            return tidy_slug(m.group(1)) + ", Australia"

    if "realestate.com.au" in host or "realcommercial.com.au" in host:
        # /property/12-smith-st-darwin-city-nt-0800
        m = re.search(r"/property/([^/?#]+)", path)
        if m and re.match(r"^\d", m.group(1)):
            return tidy_slug(m.group(1)) + ", Australia"
        # /property-house-nt-darwin+city-142... gives only suburb, useful as fallback but not exact
        m = re.search(rf"/property-[\w]+-({'|'.join(s.lower() for s in AU_STATES)})-([\w+.-]+)-\d+", path)
        if m:
            return f"{tidy_slug(m.group(2))}, {m.group(1).upper()}, Australia"

    m = re.search(rf"(\d+[a-z]?(?:-[\w']+){{1,8}}-(?:{STREET_WORDS})(?:-[\w']+){{0,8}})", path)
    if m:
        return tidy_slug(m.group(1))
    return None


def fetch_page_text(url: str, scraper_key: str = "") -> tuple[str, str]:
    methods = []
    methods.append(("direct", url, {}, 20))
    methods.append(("jina-reader", f"https://r.jina.ai/http://r.jina.ai/http://example.com".replace("http://r.jina.ai/http://example.com", url), {"X-Return-Format": "text"}, 30))
    if scraper_key:
        methods.append(("scraperapi", "https://api.scraperapi.com/", {"api_key": scraper_key, "url": url, "render": "true"}, 60))

    for name, endpoint, params, timeout in methods:
        try:
            if name == "scraperapi":
                r = requests.get(endpoint, params=params, headers=UA, timeout=timeout)
            else:
                r = requests.get(endpoint, headers=UA | params, timeout=timeout)
            if r.status_code == 200 and len(r.text) > 300:
                text = r.text if name == "jina-reader" else clean_text(r.text)
                if "captcha" not in text[:2500].lower() and len(text) > 250:
                    return text[:20000], name
        except Exception:
            continue
    return "", "failed"


def gemini_call(prompt: str, api_key: str, model: str, temperature: float = 0.1) -> str:
    if not api_key:
        return ""
    try:
        r = requests.post(
            GEMINI_URL.format(model=model),
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": 1200},
            },
            timeout=45,
        )
        data = r.json()
        if "error" in data:
            return ""
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""


def regex_extract(text: str) -> dict:
    out = {"address": None, "property_type": None, "lot_size_m2": None, "building_size_m2": None, "listed_price": None, "notes": None}
    m = re.search(r"(?:land|lot)\s*(?:size|area)?[:\s-]*([\d,.]+)\s*(m2|m²|sqm|sq\.?\s*m|acres?|ha|sq\.?\s*ft|sqft)", text, re.I)
    if m:
        val = float(m.group(1).replace(",", "")); unit = m.group(2).lower()
        if "ac" in unit: val /= M2_TO_ACRE
        elif unit == "ha": val *= 10000
        elif "ft" in unit: val /= M2_TO_SQFT
        out["lot_size_m2"] = val
    m = re.search(r"\d{1,5}[A-Za-z]?\s+[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,5}\s+(?:St|Street|Rd|Road|Ave|Avenue|Dr|Drive|Ct|Court|Cres|Crescent|Blvd|Boulevard|Ln|Lane|Way|Pl|Place|Tce|Terrace|Hwy|Highway|Pde|Parade)\b[^\n]{0,80}", text)
    if m:
        out["address"] = m.group(0).strip(" ,.-")
    return out


def llm_extract_listing(text: str, url: str, api_key: str, model: str) -> dict:
    prompt = f"""Extract real-estate/POI listing data from this page. Return ONLY valid JSON with keys:
address, property_type, lot_size_m2, building_size_m2, listed_price, notes.
Address must be full and geocodable, including suburb/city, state and country.
Convert acres to m2 by 4046.86, hectares by 10000, sqft by /10.7639. Use null when unknown.
URL: {url}
TEXT:\n{text[:13000]}"""
    raw = gemini_call(prompt, api_key, model)
    if raw:
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        m = re.search(r"\{.*\}", raw, flags=re.S)
        try:
            return json.loads(m.group(0) if m else raw)
        except Exception:
            pass
    return regex_extract(text)


def clean_address(a: str) -> str:
    a = re.sub(r"(?i)\b(for sale|sold|auction|offers?|real estate|property)\b.*", "", a)
    a = re.sub(r"(?i)^\s*(unit|apt|apartment|suite|shop|level)\s*\w+\s*[/,-]?\s*", "", a)
    a = re.sub(r"^\s*\w+/(\d+)", r"\1", a)
    return re.sub(r"\s+", " ", a).strip(" ,-|·")


def nominatim_search(q: str) -> dict | None:
    try:
        r = requests.get(f"{NOMINATIM}/search", params={"q": q, "format": "json", "limit": 1, "polygon_geojson": 1, "addressdetails": 1}, headers=UA, timeout=25)
        if r.status_code == 200:
            data = r.json()
            return data[0] if data else None
    except Exception:
        return None
    return None


def photon_search(q: str) -> dict | None:
    try:
        r = requests.get(PHOTON, params={"q": q, "limit": 1}, headers=UA, timeout=25)
        feats = r.json().get("features", [])
        if feats:
            f = feats[0]; lon, lat = f["geometry"]["coordinates"]; props = f.get("properties", {})
            label = ", ".join(str(props[k]) for k in ["name", "street", "housenumber", "city", "state", "country"] if props.get(k))
            return {"lat": lat, "lon": lon, "display_name": label or q, "geojson": None}
    except Exception:
        return None
    return None


def geocode(q: str, api_key: str, model: str, log: list[str]) -> dict | None:
    candidates = [q, clean_address(q)]
    parts = [p.strip() for p in clean_address(q).split(",") if p.strip()]
    if len(parts) >= 2:
        candidates.append(", ".join(parts[-3:]))
    seen = set()
    for cand in candidates:
        if not cand or cand.lower() in seen:
            continue
        seen.add(cand.lower())
        log.append(f"Geocoding: {cand}")
        res = nominatim_search(cand)
        if res:
            return res
        time.sleep(1.05)
        res = photon_search(cand)
        if res:
            return res
    if api_key:
        norm = gemini_call(f"Clean this into one geocodable address only: {q}", api_key, model).strip()
        if norm and norm.lower() not in seen:
            log.append(f"Geocoding LLM-normalised: {norm}")
            return nominatim_search(norm) or photon_search(norm)
    return None


def reverse_geocode(lat: float, lon: float) -> str:
    try:
        r = requests.get(f"{NOMINATIM}/reverse", params={"lat": lat, "lon": lon, "format": "json", "zoom": 18, "addressdetails": 1}, headers=UA, timeout=20)
        return r.json().get("display_name", f"{lat}, {lon}")
    except Exception:
        return f"{lat}, {lon}"


def overpass_query(q: str) -> dict:
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(endpoint, data={"data": q}, headers=UA, timeout=60)
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
    return {"elements": []}


def coords_to_poly(coords) -> Polygon | None:
    if len(coords) >= 4:
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        p = Polygon(coords)
        if p.is_valid and p.area > 0:
            return p
    return None


def fetch_osm_polygons(lat: float, lon: float, radius: int) -> dict:
    q = f"""
    [out:json][timeout:50];
    (
      way(around:{radius},{lat},{lon})["building"];
      relation(around:{radius},{lat},{lon})["building"];
      way(around:{radius},{lat},{lon})["landuse"];
      way(around:{radius},{lat},{lon})["amenity"];
      way(around:{radius},{lat},{lon})["leisure"];
      way(around:{radius},{lat},{lon})["shop"];
      way(around:{radius},{lat},{lon})["tourism"];
      relation(around:{radius},{lat},{lon})["landuse"];
      relation(around:{radius},{lat},{lon})["amenity"];
      relation(around:{radius},{lat},{lon})["shop"];
    );
    out geom tags;
    """
    data = overpass_query(q)
    pt = Point(lon, lat)
    parents, children = [], []

    for el in data.get("elements", []):
        tags = el.get("tags", {})
        poly = None
        if el.get("type") == "way" and el.get("geometry"):
            poly = coords_to_poly([(g["lon"], g["lat"]) for g in el["geometry"]])
        elif el.get("type") == "relation" and el.get("members"):
            rings = []
            for mem in el["members"]:
                if mem.get("role") in ("outer", "") and mem.get("geometry"):
                    p = coords_to_poly([(g["lon"], g["lat"]) for g in mem["geometry"]])
                    if p:
                        rings.append(p)
            if rings:
                poly = unary_union(rings)
                if poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda g: g.area)
        if poly is None or not poly.is_valid:
            continue
        if "building" in tags:
            children.append((poly, tags))
        elif any(k in tags for k in PARENT_KEYS) and (poly.contains(pt) or poly.distance(pt) < 0.0008):
            parents.append((poly, tags))
    return {"parents": parents, "children": children}


def synthetic_parent(lat: float, lon: float, lot_size_m2: float | None, radius: int) -> tuple[Polygon, dict]:
    side = math.sqrt(float(lot_size_m2 or (radius * radius)))
    d_lat = side / 2 / 111_320
    d_lon = side / 2 / (111_320 * max(math.cos(math.radians(lat)), 0.2))
    poly = Polygon([(lon-d_lon, lat-d_lat), (lon+d_lon, lat-d_lat), (lon+d_lon, lat+d_lat), (lon-d_lon, lat+d_lat), (lon-d_lon, lat-d_lat)])
    return poly, {"source": "approximate square from listed lot size/search radius"}


def run_pipeline(user_query: str, api_key: str, model: str, radius: int, progress, scraper_key: str = "") -> dict:
    res = {"source": user_query, "address": None, "lat": None, "lon": None, "listing": {}, "rows": [], "geojson_parent": [], "geojson_children": [], "log": [], "fallback_used": False}
    coords = extract_latlon_from_url_or_text(user_query)
    listing = {}

    if user_query.lower().startswith("http"):
        url_addr = address_from_url(user_query)
        if url_addr:
            listing["address"] = url_addr
            res["log"].append(f"Address parsed from URL: {url_addr}")
        progress.write("Reading page text / fallback reader…")
        text, method = fetch_page_text(user_query, scraper_key)
        res["log"].append(f"Page fetch method: {method}")
        if text:
            extracted = llm_extract_listing(text, user_query, api_key, model)
            if listing.get("address") and not extracted.get("address"):
                extracted["address"] = listing["address"]
            listing.update({k: v for k, v in extracted.items() if v not in (None, "")})
        elif coords:
            res["log"].append("Page blocked, but coordinates were found in the URL.")
            res["fallback_used"] = True
    else:
        listing["address"] = user_query

    if coords:
        lat, lon = coords
        address = reverse_geocode(lat, lon)
        res.update(lat=lat, lon=lon, address=address)
        if not listing.get("address"):
            listing["address"] = address
    else:
        addr = listing.get("address")
        if not addr:
            res["log"].append("Could not find an address or coordinates. Paste the address directly for blocked listings.")
            return res
        progress.write(f"Geocoding: {addr}")
        geo = geocode(addr, api_key, model, res["log"])
        if not geo:
            res["log"].append("Geocoding failed. Try a full address with suburb/state/country.")
            return res
        res.update(lat=float(geo["lat"]), lon=float(geo["lon"]), address=geo.get("display_name", addr))
        if geo.get("geojson"):
            try:
                g = shape(geo["geojson"])
                if g.geom_type == "MultiPolygon":
                    g = max(g.geoms, key=lambda p: p.area)
                listing["_nominatim_parent"] = g
            except Exception:
                pass

    res["listing"] = listing
    progress.write("Fetching OSM parent/child polygons…")
    osm = fetch_osm_polygons(res["lat"], res["lon"], radius)

    parent_poly, parent_tags = None, {}
    if osm["parents"]:
        # avoid accidentally selecting suburb-scale polygons when a smaller parent exists
        osm_parents = sorted(osm["parents"], key=lambda x: geodesic_area_m2(x[0]))
        sensible = [p for p in osm_parents if geodesic_area_m2(p[0]) <= 2_000_000]
        parent_poly, parent_tags = (sensible[0] if sensible else osm_parents[0])
    elif listing.get("_nominatim_parent") is not None:
        parent_poly, parent_tags = listing["_nominatim_parent"], {"source": "Nominatim polygon"}
    else:
        parent_poly, parent_tags = synthetic_parent(res["lat"], res["lon"], listing.get("lot_size_m2"), radius)
        res["fallback_used"] = True

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    parent_area = geodesic_area_m2(parent_poly)
    res["geojson_parent"].append(json.loads(json.dumps(parent_poly.__geo_interface__)))
    res["rows"].append({
        "timestamp": timestamp, "polygon_type": "PARENT", "name": parent_tags.get("name", parent_tags.get("source", parent_tags.get("landuse", "site/lot"))),
        "area_m2": round(parent_area, 1), "area_sqft": round(parent_area*M2_TO_SQFT, 0), "area_acres": round(parent_area*M2_TO_ACRE, 4),
        "address": res["address"], "lat": res["lat"], "lon": res["lon"], "source_url": user_query,
        "listed_lot_m2": listing.get("lot_size_m2"), "osm_tags": json.dumps(parent_tags)[:300], "geometry_wkt": geom_to_wkt(parent_poly),
    })

    child_count = 0
    for poly, tags in osm["children"]:
        if not poly.intersects(parent_poly.buffer(0.00025)):
            continue
        area = geodesic_area_m2(poly)
        if area < 8:
            continue
        child_count += 1
        res["geojson_children"].append(json.loads(json.dumps(poly.__geo_interface__)))
        res["rows"].append({
            "timestamp": timestamp, "polygon_type": "CHILD", "name": tags.get("name", tags.get("building", f"building_{child_count}")),
            "area_m2": round(area, 1), "area_sqft": round(area*M2_TO_SQFT, 0), "area_acres": round(area*M2_TO_ACRE, 4),
            "address": res["address"], "lat": res["lat"], "lon": res["lon"], "source_url": user_query,
            "listed_lot_m2": None, "osm_tags": json.dumps(tags)[:300], "geometry_wkt": geom_to_wkt(poly),
        })
    res["log"].append(f"Parent found: yes | Child buildings found: {child_count}")
    return res


def build_map(res: dict) -> folium.Map:
    m = folium.Map(location=[res["lat"], res["lon"]], zoom_start=18, tiles="OpenStreetMap")
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri World Imagery", name="Satellite").add_to(m)
    for gj in res["geojson_parent"]:
        folium.GeoJson(gj, name="Parent polygon", style_function=lambda _: {"color":"#4A9EFF", "weight":3, "dashArray":"6 5", "fillColor":"#4A9EFF", "fillOpacity":0.08}, tooltip="PARENT").add_to(m)
    for i, gj in enumerate(res["geojson_children"], 1):
        folium.GeoJson(gj, name=f"Child {i}", style_function=lambda _: {"color":"#FF5D5D", "weight":2, "fillColor":"#FF5D5D", "fillOpacity":0.32}, tooltip=f"CHILD {i}").add_to(m)
    folium.Marker([res["lat"], res["lon"]], tooltip=res["address"]).add_to(m)
    folium.LayerControl().add_to(m)
    return m


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="POI_Polygons")
    return buf.getvalue()


def append_to_existing_excel(upload, df_new: pd.DataFrame) -> bytes:
    try:
        old = pd.read_excel(upload)
        out = pd.concat([old, df_new], ignore_index=True)
    except Exception:
        out = df_new
    return df_to_excel_bytes(out)


st.set_page_config(page_title="POI Polygon Extractor", page_icon="🗺️", layout="wide")
st.markdown("""
<style>
.stApp {background:#0B1220;color:#E8EDF5;} 
.hero{border:1px solid #233450;border-radius:16px;padding:24px;background:linear-gradient(135deg,#0E1A30,#13233E);margin-bottom:18px;}
.hero h1{margin:0;font-size:1.8rem}.hero p{color:#9AA8BD}.tag{font-size:.75rem;border:1px solid #4A9EFF;color:#4A9EFF;border-radius:6px;padding:2px 8px;}
</style>
<div class="hero"><h1>🗺️ POI Polygon Extractor <span class="tag">EXPERT MODE</span></h1>
<p>Paste a property, mall, POI, Google Maps link, lat/lon, or plain address. The app finds parent/site polygons and child building footprints, measures areas, maps them, and exports Excel.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    default_key = get_secret("GEMINI_API_KEY", "")
    default_scraper = get_secret("SCRAPERAPI_KEY", "")
    api_key = st.text_input("Gemini API key", value=default_key, type="password", help="Optional, but improves extraction from messy pages.")
    scraper_key = st.text_input("ScraperAPI key", value=default_scraper, type="password", help="Optional fallback for blocked pages.")
    model = st.selectbox("Gemini model", ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"])
    radius = st.slider("OSM search radius (metres)", 40, 800, 160, 20)
    existing_xlsx = st.file_uploader("Append to existing Excel", type=["xlsx"])
    debug = st.toggle("Show debug log", value=True)
    if st.button("Clear session"):
        st.session_state.clear(); st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant", "content":"Paste a POI/property link or address and I’ll extract parent + child polygons."}]
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "last_result" not in st.session_state:
    st.session_state.last_result = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

URL_RE = re.compile(r"https?://\S+")
ADDRESS_RE = re.compile(r"\d+\s+.+\b(street|st\b|road|rd\b|ave|avenue|drive|dr\b|court|ct\b|cres|blvd|lane|ln\b|way|place|pl\b|terrace|tce|highway|hwy)\b", re.I)
LATLON_RE = re.compile(r"-?\d{1,2}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}")

user_input = st.chat_input("Paste link, address, or lat/lon…")
if user_input:
    st.session_state.messages.append({"role":"user", "content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        urls = URL_RE.findall(user_input)
        is_extract = bool(urls or ADDRESS_RE.search(user_input) or LATLON_RE.search(user_input))
        if is_extract:
            query = urls[0] if urls else user_input.strip()
            with st.status("Extracting polygons…", expanded=True) as progress:
                result = run_pipeline(query, api_key, model, radius, progress, scraper_key)
                progress.update(label="Extraction complete", state="complete", expanded=False)
            if result["rows"]:
                df_new = pd.DataFrame(result["rows"])
                st.session_state.df = pd.concat([st.session_state.df, df_new], ignore_index=True)
                st.session_state.last_result = result
                parent = df_new[df_new["polygon_type"] == "PARENT"].iloc[0]
                children = len(df_new[df_new["polygon_type"] == "CHILD"])
                msg = f"Extracted **{len(df_new)} polygon(s)** for **{result['address']}**.\n\n🟦 Parent: {fmt_area(parent.area_m2)}\n\n🟥 Child buildings: **{children}**"
                if result["fallback_used"]:
                    msg += "\n\nNote: fallback mode was used, so parent boundary may be approximate if OSM has no exact site polygon."
                st.markdown(msg)
                if debug:
                    with st.expander("Debug log"):
                        st.write("\n".join(result["log"]))
                st.session_state.messages.append({"role":"assistant", "content":msg})
            else:
                msg = "I could not extract polygons. Try pasting the full street address or a Google Maps link with coordinates."
                st.error(msg)
                if debug:
                    st.write("\n".join(result["log"]))
                st.session_state.messages.append({"role":"assistant", "content":msg})
        else:
            ctx = st.session_state.df.to_csv(index=False)[:7000] if not st.session_state.df.empty else "No polygon data yet."
            answer = gemini_call(f"You are a GIS POI assistant. Data:\n{ctx}\n\nQuestion: {user_input}", api_key, model, temperature=0.5) or "Paste a property/POI link or address first. Add a Gemini API key for data Q&A."
            st.markdown(answer)
            st.session_state.messages.append({"role":"assistant", "content":answer})

if not st.session_state.df.empty:
    st.divider()
    tab1, tab2, tab3 = st.tabs(["Map", "Data", "Export"])
    with tab1:
        if st.session_state.last_result:
            st_folium(build_map(st.session_state.last_result), height=560, use_container_width=True)
    with tab2:
        st.dataframe(st.session_state.df, use_container_width=True)
    with tab3:
        c1, c2, c3 = st.columns(3)
        c1.download_button("Download CSV", st.session_state.df.to_csv(index=False).encode("utf-8"), "poi_polygons.csv", "text/csv", use_container_width=True)
        c2.download_button("Download Excel", df_to_excel_bytes(st.session_state.df), "poi_polygons.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        if existing_xlsx is not None:
            c3.download_button("Download merged Excel", append_to_existing_excel(existing_xlsx, st.session_state.df), f"merged_{existing_xlsx.name}", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        else:
            c3.info("Upload an .xlsx in the sidebar to merge.")

st.caption("Data © OpenStreetMap contributors. Geocoding: Nominatim/Photon. Polygons: Overpass API. LLM: optional Gemini API.")
