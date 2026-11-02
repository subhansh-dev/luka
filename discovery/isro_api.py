"""ISRO API wrapper — MOSDAC + Bhuvan.

Real implementation using MOSDAC and Bhuvan public APIs.
MOSDAC: https://mosdac.gov.in/
Bhuvan: https://bhuvan-app1.nrsc.gov.in/
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

API_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
MOSDAC_BASE = "https://mosdac.gov.in/api"
BHUVAN_BASE = "https://bhuvan-app1.nrsc.gov.in/api"
_LAST_CALL = 0.0


def _rate_limit():
    global _LAST_CALL
    now = time.time()
    if now - _LAST_CALL < 1.5:
        time.sleep(1.5 - (now - _LAST_CALL))
    _LAST_CALL = time.time()


def _get_config() -> dict:
    try:
        return json.loads(API_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _fetch_json(url: str, headers: dict = None) -> dict | None:
    _rate_limit()
    h = {"User-Agent": "Luka-Space-Discovery/1.0"}
    if headers:
        h.update(headers)
    try:
        req = urllib.request.Request(url, headers=h)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _fetch_text(url: str) -> str | None:
    _rate_limit()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Luka-Space-Discovery/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode()
    except Exception:
        return None


# --- MOSDAC Functions ---

def search_satellite_data(query: str, mission: str = "", limit: int = 5) -> list[dict]:
    """Search MOSDAC satellite data catalog.

    Queries the MOSDAC catalog for satellite data products.
    Falls back to known ISRO mission metadata if API is unreachable.
    """
    # Try real MOSDAC search
    safe_query = urllib.parse.quote(query)
    url = f"{MOSDAC_BASE}/catalog/search?q={safe_query}&limit={limit}"
    if mission:
        url += f"&mission={urllib.parse.quote(mission)}"
    data = _fetch_json(url)

    if data and isinstance(data, list):
        return [{
            "mission": item.get("mission", mission),
            "product": item.get("product_name", ""),
            "description": item.get("description", ""),
            "date": item.get("start_date", ""),
            "coverage": item.get("spatial_coverage", ""),
            "source": "MOSDAC",
        } for item in data[:limit]]

    # Fallback: known ISRO missions with real metadata
    ISRO_MISSIONS = {
        "insat-3dr": {"full_name": "INSAT-3DR", "type": "Meteorological", "instruments": ["IMAGER", "SOUNDER"], "orbit": "GEO 74°E", "launch": "2016"},
        "insat-3d": {"full_name": "INSAT-3D", "type": "Meteorological", "instruments": ["IMAGER", "SOUNDER"], "orbit": "GEO 82°E", "launch": "2013"},
        "oceansat-2": {"full_name": "OCEANSAT-2", "type": "Ocean observation", "instruments": ["OCM-2", "SCAT"], "orbit": "SSO 720km", "launch": "2009"},
        "oceansat-3": {"full_name": "OCEANSAT-3 (EOS-06)", "type": "Ocean observation", "instruments": ["OCM-3", "SCAT"], "orbit": "SSO 720km", "launch": "2022"},
        "scatsat-1": {"full_name": "SCATSAT-1", "type": "Ocean wind", "instruments": ["SCAT"], "orbit": "SSO 720km", "launch": "2016"},
        "risat-1": {"full_name": "RISAT-1", "type": "Radar imaging", "instruments": ["C-SAR"], "orbit": "SSO 536km", "launch": "2012"},
        "risat-2br1": {"full_name": "RISAT-2BR1", "type": "Radar imaging", "instruments": ["X-SAR"], "orbit": "SSO 557km", "launch": "2019"},
        "cartosat-3": {"full_name": "CARTOSAT-3", "type": "High-res imaging", "instruments": ["PAN", "MX"], "orbit": "SSO 509km", "launch": "2019"},
        "resourcesat-2a": {"full_name": "RESOURCESAT-2A", "type": "Earth resources", "instruments": ["LISS-4", "LISS-3", "AWiFS"], "orbit": "SSO 817km", "launch": "2016"},
        "saphir": {"full_name": "SAPHIR (on Megha-Tropiques)", "type": "Atmospheric", "instruments": ["SAPHIR"], "orbit": "SSO 867km", "launch": "2011"},
    }

    query_lower = query.lower().replace(" ", "").replace("-", "")
    for key, info in ISRO_MISSIONS.items():
        if key.replace("-", "") in query_lower or query_lower in key.replace("-", ""):
            return [{
                "mission": info["full_name"],
                "product": info["type"],
                "instruments": info["instruments"],
                "orbit": info["orbit"],
                "launch_year": info["launch"],
                "source": "ISRO (known mission)",
            }]

    return [{"mission": mission or query, "product": "Not found in MOSDAC catalog", "source": "MOSDAC (unverified)"}]


def get_meteorological_data(lat: float, lon: float, parameter: str = "rainfall") -> dict:
    """Get meteorological data from ISRO satellites.

    Queries MOSDAC for real-time or archival meteorological data.
    Returns real parameter metadata with coordinates.
    """
    # Try MOSDAC API
    url = f"{MOSDAC_BASE}/data/meteorological?lat={lat}&lon={lon}&param={urllib.parse.quote(parameter)}"
    data = _fetch_json(url)

    if data and isinstance(data, dict) and data.get("value") is not None:
        return {
            "lat": lat, "lon": lon,
            "parameter": parameter,
            "value": data["value"],
            "unit": data.get("unit", "unknown"),
            "timestamp": data.get("timestamp", ""),
            "source": "MOSDAC",
        }

    # Return metadata about what MOSDAC provides for this location
    # India bounds: roughly 6°N-35°N, 68°E-97°E
    in_india = 6 <= lat <= 35 and 68 <= lon <= 97

    return {
        "lat": lat, "lon": lon,
        "parameter": parameter,
        "value": None,
        "status": "API unreachable" if in_india else "Outside India coverage",
        "coverage": "INSAT-3DR/3D series" if in_india else "N/A — ISRO covers Indian subcontinent",
        "note": "Real MOSDAC API requires registration for data access",
        "source": "MOSDAC (metadata only)",
    }


def get_ocean_data(lat: float, lon: float, parameter: str = "sst") -> dict:
    """Get ocean data from OCEANSAT satellites.

    Args:
        lat, lon: Coordinates
        parameter: "sst" (sea surface temperature), "chlorophyll", "wind_speed"

    Returns:
        Dict with parameter metadata.
    """
    url = f"{MOSDAC_BASE}/data/ocean?lat={lat}&lon={lon}&param={urllib.parse.quote(parameter)}"
    data = _fetch_json(url)

    if data and isinstance(data, dict) and data.get("value") is not None:
        return {
            "lat": lat, "lon": lon,
            "parameter": parameter,
            "value": data["value"],
            "unit": data.get("unit", ""),
            "source": "MOSDAC/OCEANSAT",
        }

    # Ocean coverage is global for OCEANSAT
    PARAM_UNITS = {"sst": "°C", "chlorophyll": "mg/m³", "wind_speed": "m/s", "salinity": "PSU"}

    return {
        "lat": lat, "lon": lon,
        "parameter": parameter,
        "value": None,
        "unit": PARAM_UNITS.get(parameter, "unknown"),
        "status": "API unreachable — OCEANSAT-3/SCATSAT-1 coverage is global",
        "source": "MOSDAC (metadata only)",
    }


# --- Bhuvan Functions ---

def geocode_village(name: str) -> dict | None:
    """Geocode Indian village/city name to lat/lon via Bhuvan.

    Uses Bhuvan's gazetteer service.
    """
    safe_name = urllib.parse.quote(name)
    url = f"{BHUVAN_BASE}/gazetteer/search?name={safe_name}"
    data = _fetch_json(url)

    if data and isinstance(data, list) and len(data) > 0:
        return {
            "village": data[0].get("name", name),
            "lat": data[0].get("lat"),
            "lon": data[0].get("lon"),
            "state": data[0].get("state", ""),
            "district": data[0].get("district", ""),
            "source": "Bhuvan Gazetteer",
        }

    # Known Indian cities as fallback
    CITIES = {
        "chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
        "mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
        "delhi": {"lat": 28.7041, "lon": 77.1025, "state": "Delhi"},
        "bangalore": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
        "bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
        "hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
        "kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
        "ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat"},
        "pune": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra"},
        "jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
        "sriharikota": {"lat": 13.7199, "lon": 80.2304, "state": "Andhra Pradesh"},
        "isro_hq": {"lat": 12.9516, "lon": 77.5946, "state": "Karnataka"},
    }

    name_lower = name.lower().strip()
    for city, info in CITIES.items():
        if city in name_lower or name_lower in city:
            return {
                "village": name,
                "lat": info["lat"],
                "lon": info["lon"],
                "state": info["state"],
                "source": "Luka geocoding (known city)",
            }

    return {
        "village": name,
        "lat": None,
        "lon": None,
        "status": "Not found — Bhuvan API unreachable and not in known cities",
        "source": "Bhuvan (unresolved)",
    }


def get_lulc_stats(lat: float, lon: float, radius_km: float = 10, scale: str = "50K") -> dict:
    """Get Land Use/Land Cover statistics from Bhuvan.

    Args:
        lat, lon: Center coordinates
        radius_km: Analysis radius
        scale: Map scale ("50K", "250K", "1M")

    Returns:
        Dict with land use/land cover statistics.
    """
    url = f"{BHUVAN_BASE}/lulc/stats?lat={lat}&lon={lon}&radius={radius_km}&scale={scale}"
    data = _fetch_json(url)

    if data and isinstance(data, dict) and data.get("stats"):
        return {
            "lat": lat, "lon": lon,
            "scale": scale,
            "radius_km": radius_km,
            "stats": data["stats"],
            "source": "Bhuvan LULC",
        }

    return {
        "lat": lat, "lon": lon,
        "scale": scale,
        "radius_km": radius_km,
        "stats": None,
        "status": "Bhuvan LULC API unreachable",
        "note": "Bhuvan LULC requires specific WMS/WFS queries to nrsc.gov.in",
        "source": "Bhuvan (metadata only)",
    }


def is_available() -> bool:
    """Check if ISRO APIs are reachable."""
    cfg = _get_config()
    if cfg.get("mosdac_api_key"):
        return True
    # Test MOSDAC homepage
    try:
        req = urllib.request.Request("https://mosdac.gov.in/", headers={"User-Agent": "Luka/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


def enrich_entities(graph, entity_types: set[str] = {"satellite", "instrument", "region", "parameter"}) -> int:
    """Enrich knowledge graph entities with ISRO data."""
    enriched = 0
    for eid, ent in list(graph.entities.items()):
        if ent.get("type") not in entity_types:
            continue
        name = ent.get("name", "")
        if not name:
            continue

        if ent["type"] == "satellite":
            data = search_satellite_data(name, mission=name)
            if data:
                graph.entities[eid].setdefault("isro", {})["mission_info"] = data[0]
                enriched += 1
        elif ent["type"] == "parameter":
            if any(k in name.lower() for k in ("temp", "sst", "sea surface")):
                data = get_ocean_data(20.0, 70.0, "sst")
                graph.entities[eid].setdefault("isro", {})["measurement"] = data
                enriched += 1
            elif any(k in name.lower() for k in ("rain", "precip", "monsoon")):
                data = get_meteorological_data(20.0, 77.0, "rainfall")
                graph.entities[eid].setdefault("isro", {})["measurement"] = data
                enriched += 1
    return enriched
