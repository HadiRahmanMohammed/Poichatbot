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
# 0) Address straight from the URL (no fetching needed — unblockable)
#    Big portals embed the full address in the URL slug.
# ----------------------------------------------------------------------------
STREET_WORDS = (r"street|st|road|rd|avenue|ave|drive|dr|court|ct|crescent|cres|boulevard|blvd|"
                r"lane|ln|way|place|pl|terrace|tce|highway|hwy|parade|pde|circuit|cct|"
                r"esplanade|esp|circle|cir|trail|trl|parkway|pkwy|square|sq")
US_STATES = ("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT "
             "NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC").split()
AU_STATES = "NSW VIC QLD WA SA TAS ACT NT".split()


def address_from_url(url: str) -> str | None:
    """Parse the address directly out of well-known portal URL patterns."""
    from urllib.parse import urlparse, unquote
    u = urlparse(url.lower())
    host, path = u.netloc, unquote(u.path)

    def tidy(s: str) -> str:
        s = s.replace("+", " ").replace("_", " ").replace("-", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # Zillow: /homedetails/1234-N-Main-St-Phoenix-AZ-85001/12345_zpid/
    if "zillow." in host:
        m = re.search(r"/homedetails/([^/]+)/", path)
        if m:
            slug = tidy(m.group(1))
            m2 = re.search(rf"^(.*?)\s({'|'.join(s.lower() for s in US_STATES)})\s(\d{{5}})$", slug)
            if m2:
                return f"{m2.group(1)}, {m2.group(2).upper()} {m2.group(3)}, USA"
            return slug + ", USA"

    # Redfin: /AZ/Phoenix/1234-N-Main-St-85001/home/123456
    if "redfin." in host:
        m = re.search(r"/([a-z]{2})/([^/]+)/([^/]+)/home/", path)
        if m:
            state, city, street = m.group(1).upper(), tidy(m.group(2)), tidy(m.group(3))
            street = re.sub(r"\s\d{5}$", "", street)
            return f"{street}, {city}, {state}, USA"

    # Realtor.com: /realestateandhomes-detail/1234-Main-St_Phoenix_AZ_85001_M123-456
    if "realtor.com" in host:
        m = re.search(r"/realestateandhomes-detail/([^/]+)", path)
        if m:
            parts = m.group(1).split("_")
            parts = [p for p in parts if not re.match(r"^m\d", p)]
            return tidy(", ".join(parts)) + ", USA"

    # Trulia: /p/az/phoenix/1234-n-main-st-phoenix-az-85001--123456
    if "trulia." in host:
        m = re.search(r"/p/[a-z]{2}/[^/]+/([^/]+?)(?:--\d+)?/?$", path)
        if m:
            return tidy(m.group(1)) + ", USA"

    # Domain.com.au: /12-smith-street-darwin-city-nt-0800-2019001234
    if "domain.com.au" in host:
        m = re.search(r"/(\d+[a-z]?(?:-[\w']+)+?-(?:%s)-[\w-]+?-(?:%s)-\d{4})" %
                      (STREET_WORDS, "|".join(s.lower() for s in AU_STATES)), path)
        if m:
            return tidy(m.group(1)) + ", Australia"
        m = re.search(r"/([\w'+-]+)-(\d{4})-\d{6,}", path)  # suburb-postcode-id
        if m:
            return f"{tidy(m.group(1))} {m.group(2)}, Australia"

    # realestate.com.au: /property/12-smith-st-darwin-city-nt-0800/  OR
    #                    /property-house-nt-darwin+city-141234567
    if "realestate.com" in host:
        m = re.search(r"/property/([\w'+-]+?)/?$", path)
        if m and re.match(r"^\d", m.group(1)):
            return tidy(m.group(1)) + ", Australia"
        m = re.search(r"/property-[\w]+-(%s)-([\w'+]+)-\d{6,}" %
                      "|".join(s.lower() for s in AU_STATES), path)
        if m:
            return f"{tidy(m.group(2))}, {m.group(1).upper()}, Australia"

    # Generic: number-street-words-...-(state)-(zip/postcode) anywhere in path
    m = re.search(rf"(\d+[a-z]?(?:-[\w']+){{1,6}}-(?:{STREET_WORDS})(?:-[\w']+){{0,5}})", path)
    if m:
        addr = tidy(m.group(1))
        tail = re.search(rf"\b({'|'.join(s.lower() for s in US_STATES + AU_STATES)})[-/](\d{{4,5}})\b", path)
        if tail:
            addr += f", {tail.group(1).upper()} {tail.group(2)}"
        return addr
    return None


# ----------------------------------------------------------------------------
# 1) Page fetching (direct -> Jina reader -> CORS proxy -> ScraperAPI)
# ----------------------------------------------------------------------------
def fetch_page_text(url: str, scraper_key: str = "") -> tuple[str, str]:
    """Return (text, method). Cascades through free fetch strategies."""

    def strip_html(html: str) -> str:
        t = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
        t = re.sub(r"<style.*?</style>", " ", t, flags=re.S | re.I)
        t = re.sub(r"<[^>]+>", " ", t)
        return re.sub(r"\s+", " ", t)

    # 1. Direct with realistic browser headers
    try:
        r = requests.get(url, headers={**UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.google.com/", "DNT": "1",
            "Upgrade-Insecure-Requests": "1"}, timeout=15)
        if r.status_code == 200 and len(r.text) > 500:
            txt = strip_html(r.text)
            if len(txt) > 400 and "captcha" not in txt[:2000].lower():
                return txt[:18000], "direct"
    except Exception:
        pass

    # 2. Jina AI reader (free, renders JS)
    try:
        r = requests.get(f"https://r.jina.ai/{url}",
                         headers={**UA, "X-Return-Format": "text"}, timeout=25)
        if r.status_code == 200 and len(r.text) > 300 and "captcha" not in r.text[:2000].lower():
            return r.text[:18000], "jina-reader"
    except Exception:
        pass

    # 3. AllOrigins CORS proxy (free)
    try:
        r = requests.get("https://api.allorigins.win/raw",
                         params={"url": url}, headers=UA, timeout=25)
        if r.status_code == 200 and len(r.text) > 500:
            txt = strip_html(r.text)
            if len(txt) > 400:
                return txt[:18000], "allorigins-proxy"
    except Exception:
        pass

    # 4. ScraperAPI (optional user key — free tier 1000 req/mo, handles JS + blocks)
    if scraper_key:
        try:
            r = requests.get("https://api.scraperapi.com/",
                             params={"api_key": scraper_key, "url": url, "render": "true"},
                             timeout=60)
            if r.status_code == 200 and len(r.text) > 500:
                return strip_html(r.text)[:18000], "scraperapi"
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
  "address": FULL geocodable street address as one string — must include suburb/city, state and country; infer the country from the website domain if needed (or null),
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
def clean_address(addr: str) -> str:
    """Strip listing noise Nominatim chokes on: units, lots, prices, marketing text."""
    a = addr.strip()
    a = re.sub(r"(?i)\b(?:unit|apt|apartment|suite|shop|level)\s*\d+\w?\s*[,/]?\s*", "", a)
    a = re.sub(r"(?i)\blot\s*\d+\w?\s*[,/]?\s*", "", a)
    a = re.sub(r"^\d+\w?/(\d)", r"\1", a)          # "2/45 Smith St" -> "45 Smith St"
    a = re.sub(r"(?i)\$[\d,\.]+[km]?", "", a)       # prices
    a = re.sub(r"(?i)\b(for sale|for rent|sold|auction|offers?)\b.*", "", a)
    a = re.sub(r"\s+", " ", a).strip(" ,-|·")
    return a


def _nominatim(q: str) -> dict | None:
    try:
        r = requests.get(
            f"{NOMINATIM}/search",
            params={"q": q, "format": "json", "limit": 1,
                    "polygon_geojson": 1, "addressdetails": 1},
            headers=UA, timeout=20,
        )
        res = r.json()
        if res:
            return res[0]
    except Exception:
        pass
    return None


def _photon(q: str) -> dict | None:
    """Free komoot Photon geocoder — much more forgiving than Nominatim."""
    try:
        r = requests.get("https://photon.komoot.io/api/",
                         params={"q": q, "limit": 1}, headers=UA, timeout=20)
        feats = r.json().get("features", [])
        if feats:
            f = feats[0]
            lon, lat = f["geometry"]["coordinates"]
            props = f.get("properties", {})
            label = ", ".join(str(props[k]) for k in
                              ("name", "street", "housenumber", "city", "state", "country")
                              if props.get(k))
            return {"lat": lat, "lon": lon, "display_name": label or q, "geojson": None}
    except Exception:
        pass
    return None


def geocode(query: str, api_key: str = "", model: str = "gemini-2.0-flash",
            log: list | None = None) -> dict | None:
    """Multi-strategy geocoding cascade. Logs every attempt if a log list is given."""
    def _log(msg):
        if log is not None:
            log.append(msg)

    candidates = [query]
    cleaned = clean_address(query)
    if cleaned and cleaned != query:
        candidates.append(cleaned)
    # street + locality only (drop leading descriptors)
    m = re.search(r"\d+\w?\s+[\w' \-]+?(?:St|Street|Rd|Road|Ave|Avenue|Dr|Drive|Ct|Court|"
                  r"Cres|Crescent|Blvd|Boulevard|Ln|Lane|Way|Pl|Place|Tce|Terrace|Hwy|Highway|"
                  r"Pde|Parade|Cct|Circuit|Esp|Esplanade)\b.*", cleaned or query, re.I)
    if m:
        candidates.append(m.group(0))
    # locality-level last resort: last 2-3 comma parts
    parts = [p.strip() for p in (cleaned or query).split(",") if p.strip()]
    if len(parts) >= 2:
        candidates.append(", ".join(parts[-3:]))

    seen = set()
    for cand in candidates:
        if not cand or cand.lower() in seen:
            continue
        seen.add(cand.lower())
        _log(f"Geocode try (Nominatim): '{cand}'")
        res = _nominatim(cand)
        if res:
            _log("  ↳ ✅ hit")
            return res
        time.sleep(1.1)  # Nominatim rate policy
        _log(f"Geocode try (Photon): '{cand}'")
        res = _photon(cand)
        if res:
            _log("  ↳ ✅ hit (Photon)")
            return res

    # Final resort: ask the LLM to normalise the address, retry both geocoders
    if api_key:
        norm = gemini_call(
            f"Reformat this into a clean geocodable address in the form "
            f"'house_number street, suburb, state, country' — output ONLY the address, "
            f"nothing else. If a country is missing, infer it: '{query}'",
            api_key, model,
        ).strip()
        if norm and norm.lower() not in seen and len(norm) < 140:
            _log(f"Geocode try (LLM-normalised): '{norm}'")
            res = _nominatim(norm) or _photon(norm)
            if res:
                _log("  ↳ ✅ hit")
                return res
    _log("  ↳ ❌ all geocoding strategies failed")
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
def run_pipeline(query: str, api_key: str, model: str, radius: int, progress, scraper_key: str = "") -> dict:
    result = {
        "source": query, "address": None, "lat": None, "lon": None,
        "listing": {}, "rows": [], "geojson_parent": [], "geojson_children": [],
        "log": [], "fallback_used": False,
    }
    is_url = query.lower().startswith("http")
    listing = {}

    if is_url:
        # 0) Address straight from the URL — instant, cannot be blocked
        url_addr = address_from_url(query)
        if url_addr:
            listing["address"] = url_addr
            result["log"].append(f"Address parsed from URL slug: '{url_addr}' (no scraping needed)")

        # 1) Fetch page for enrichment (lot size, price…) — optional if we have the address
        progress.write("🌐 Fetching page…")
        text, method = fetch_page_text(query, scraper_key)
        result["log"].append(f"Page fetch: {method}")
        if text:
            progress.write(f"🤖 Extracting listing data ({'Gemini' if api_key else 'regex'})…")
            extracted = llm_extract_listing(text, query, api_key, model)
            # keep URL-parsed address unless page gives a more complete one
            if url_addr and (not extracted.get("address") or len(str(extracted.get("address"))) < len(url_addr)):
                extracted["address"] = url_addr
            listing = {**listing, **{k: v for k, v in extracted.items() if v}}
        elif url_addr:
            result["log"].append("Page blocked — proceeding with URL-parsed address + OSM polygons (cross-reference mode).")
            result["fallback_used"] = True
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

    result["log"].append(f"Address extracted: '{addr}'")
    progress.write(f"📍 Geocoding: {addr}")
    geo = geocode(addr, api_key, model, log=result["log"])
    if not geo:
        result["log"].append("❌ Geocoding failed on all strategies.")
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
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite",
    ).add_to(m)
    for gj in res["geojson_parent"]:
        folium.GeoJson(
            gj, name="Parent polygon",
            style_function=lambda x: {"color": "#4A9EFF", "weight": 3, "dashArray": "7 5",
                                      "fillColor": "#4A9EFF", "fillOpacity": 0.08},
            tooltip="PARENT · lot / site",
        ).add_to(m)
    for gj in res["geojson_children"]:
        folium.GeoJson(
            gj, name="Child polygon",
            style_function=lambda x: {"color": "#FF5D5D", "weight": 2,
                                      "fillColor": "#FF5D5D", "fillOpacity": 0.30},
            tooltip="CHILD · building",
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

# ---------------------------------------------------------------------------
# Design system — "Night Cartography"
#   ink navy canvas · survey-grid texture · UI accents = polygon legend colors
#   PARENT #4A9EFF (blue) · CHILD #FF5D5D (coral) · Space Grotesk + JetBrains Mono
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --ink:#0B1220; --panel:#131C2E; --panel-2:#182338; --line:#233450;
  --text:#E8EDF5; --muted:#8B97AB;
  --parent:#4A9EFF; --child:#FF5D5D; --ok:#3DDC97; --warn:#FFB454;
}

html, body, [class*="css"], .stApp { font-family:'Space Grotesk',sans-serif; }
.stApp { background:
  linear-gradient(rgba(74,158,255,0.035) 1px, transparent 1px),
  linear-gradient(90deg, rgba(74,158,255,0.035) 1px, transparent 1px),
  var(--ink);
  background-size: 42px 42px, 42px 42px, auto; }

/* ---------- hero ---------- */
.hero{ position:relative; overflow:hidden; border:1px solid var(--line);
  border-radius:16px; padding:26px 30px 22px;
  background:linear-gradient(135deg,#0E1A30 0%,#101C33 55%,#13233E 100%); }
.hero::after{ content:""; position:absolute; inset:0; pointer-events:none;
  background-image:
    linear-gradient(rgba(74,158,255,.07) 1px,transparent 1px),
    linear-gradient(90deg,rgba(74,158,255,.07) 1px,transparent 1px);
  background-size:34px 34px; }
.hero h1{ margin:0; font-size:1.75rem; font-weight:700; letter-spacing:.3px; color:var(--text); }
.hero h1 .mode{ font-family:'JetBrains Mono',monospace; font-size:.5em; color:var(--parent);
  border:1px solid var(--parent); border-radius:6px; padding:2px 8px; vertical-align:middle;
  margin-left:10px; letter-spacing:1px; }
.hero p{ margin:8px 0 0; color:var(--muted); max-width:720px; }
.hero .coords{ position:absolute; top:14px; right:20px; font-family:'JetBrains Mono',monospace;
  font-size:.68rem; color:var(--muted); letter-spacing:1px; opacity:.8; }
.hero svg{ position:absolute; right:26px; bottom:-6px; opacity:.9; }
.legend{ display:flex; gap:18px; margin-top:14px; font-family:'JetBrains Mono',monospace; font-size:.72rem; }
.legend span{ display:inline-flex; align-items:center; gap:7px; color:var(--muted); letter-spacing:.5px;}
.swatch{ width:12px; height:12px; border-radius:3px; display:inline-block; }
.swatch.p{ background:rgba(74,158,255,.25); border:2px solid var(--parent); }
.swatch.c{ background:rgba(255,93,93,.35); border:2px solid var(--child); }

/* ---------- chat ---------- */
[data-testid="stChatMessage"]{ background:var(--panel); border:1px solid var(--line);
  border-radius:14px; padding:14px 16px; }
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]),
[data-testid="stChatMessage"][data-testid*="user"]{ background:var(--panel-2); }
[data-testid="stChatInput"] textarea{ font-family:'Space Grotesk',sans-serif !important; }
[data-testid="stChatInput"]{ border:1px solid var(--line); border-radius:14px; }
[data-testid="stChatInput"]:focus-within{ border-color:var(--parent);
  box-shadow:0 0 0 3px rgba(74,158,255,.18); }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0D1626,#0B1220);
  border-right:1px solid var(--line); }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{
  font-size:.8rem; text-transform:uppercase; letter-spacing:2px; color:var(--muted);
  font-family:'JetBrains Mono',monospace; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"]{ gap:6px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{ font-family:'JetBrains Mono',monospace; font-size:.8rem;
  letter-spacing:1px; border-radius:10px 10px 0 0; padding:8px 18px; color:var(--muted); }
.stTabs [aria-selected="true"]{ color:var(--parent) !important;
  background:rgba(74,158,255,.08); border-bottom:2px solid var(--parent); }

/* ---------- metrics as legend cards ---------- */
[data-testid="stMetric"]{ background:var(--panel); border:1px solid var(--line);
  border-radius:14px; padding:14px 18px; }
[data-testid="stMetric"] [data-testid="stMetricLabel"]{ font-family:'JetBrains Mono',monospace;
  font-size:.68rem; letter-spacing:1.5px; text-transform:uppercase; color:var(--muted); }
[data-testid="stMetric"] [data-testid="stMetricValue"]{ font-family:'JetBrains Mono',monospace; }
div[data-testid="column"]:nth-of-type(2) [data-testid="stMetricValue"]{ color:var(--parent); }
div[data-testid="column"]:nth-of-type(3) [data-testid="stMetricValue"]{ color:var(--child); }

/* ---------- buttons ---------- */
.stDownloadButton button, .stButton button{
  border:1px solid var(--line); border-radius:12px; background:var(--panel-2);
  color:var(--text); font-family:'JetBrains Mono',monospace; font-size:.78rem;
  letter-spacing:.5px; transition:all .15s ease; }
.stDownloadButton button:hover, .stButton button:hover{
  border-color:var(--parent); color:var(--parent);
  box-shadow:0 0 0 3px rgba(74,158,255,.15); transform:translateY(-1px); }

/* ---------- dataframe / status / misc ---------- */
[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:14px; overflow:hidden; }
[data-testid="stStatus"]{ background:var(--panel); border:1px solid var(--line); border-radius:14px; }
[data-testid="stFileUploader"]{ border:1px dashed var(--line); border-radius:14px; padding:6px; }
hr{ border-color:var(--line) !important; }
code, .mono{ font-family:'JetBrains Mono',monospace; }
::-webkit-scrollbar{ width:9px; height:9px; }
::-webkit-scrollbar-thumb{ background:var(--line); border-radius:6px; }
::-webkit-scrollbar-thumb:hover{ background:var(--parent); }
@media (prefers-reduced-motion: reduce){ *{ transition:none !important; animation:none !important; } }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div class="coords">LAT −12.4634 · LON 130.8456 · WGS84</div>
      <h1>🗺️ POI Polygon Extractor <span class="mode">EXPERT&nbsp;MODE</span></h1>
      <p>Paste a real-estate / mall / POI link — I'll trace the site boundary and every building
         footprint inside it, measure true geodesic areas, and export to Excel.</p>
      <div class="legend">
        <span><i class="swatch p"></i>PARENT · LOT / SITE</span>
        <span><i class="swatch c"></i>CHILD · BUILDING</span>
      </div>
      <svg width="150" height="86" viewBox="0 0 150 86" fill="none">
        <polygon points="8,78 22,14 118,6 142,60 96,82" stroke="#4A9EFF" stroke-width="2"
                 stroke-dasharray="7 5" fill="rgba(74,158,255,0.07)"/>
        <polygon points="46,50 58,26 100,24 108,52 78,62" stroke="#FF5D5D" stroke-width="2"
                 fill="rgba(255,93,93,0.16)"/>
        <circle cx="22" cy="14" r="3.5" fill="#4A9EFF"/><circle cx="142" cy="60" r="3.5" fill="#4A9EFF"/>
        <circle cx="8" cy="78" r="3.5" fill="#4A9EFF"/><circle cx="118" cy="6" r="3.5" fill="#4A9EFF"/>
        <circle cx="96" cy="82" r="3.5" fill="#4A9EFF"/>
      </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API key (free — aistudio.google.com)", type="password")
    scraper_key = st.text_input("ScraperAPI key (optional — scraperapi.com free tier)", type="password",
                                help="Only needed if a portal blocks all free fetch methods. 1,000 free requests/month.")
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
                res = run_pipeline(query, api_key, model, radius, progress, scraper_key)
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
            st.caption("🟦 dashed = PARENT lot/site · 🟥 solid = CHILD buildings · switch to Satellite in the layer control ↗")

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
    "<p style=\"text-align:center;color:#8B97AB;font-family:'JetBrains Mono',monospace;"
    "font-size:.68rem;letter-spacing:1px;margin-top:28px;\">"
    "DATA © OPENSTREETMAP · GEOCODING NOMINATIM + PHOTON · POLYGONS OVERPASS · LLM GEMINI FREE TIER</p>",
    unsafe_allow_html=True,
)
