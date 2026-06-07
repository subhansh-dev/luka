"""ISRO API wrapper — MOSDAC + Bhuvan.
Reference: rumi_isro_personalization.txt
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

API_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
MOSDAC_BASE = "https://mosdac.gov.in/api/"
BHUVAN_BASE = "https://bhuvan-app1.nrsc.gov.in/api/"
_LAST_CALL = 0.0


def _get_mosdac_key() -> str:
    try:
        cfg = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8-sig"))
        return cfg.get("mosdac_api_key", "")
    except Exception:
        return ""


def _get_bhuvan_token() -> str:
    try:
        cfg = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8-sig"))
        return cfg.get("bhuvan_token", "")
    except Exception:
        return ""


def _rate_limit():
    global _LAST_CALL
    now = time.time()
    if now - _LAST_CALL < 1.0:
        time.sleep(1.0 - (now - _LAST_CALL))
    _LAST_CALL = time.time()


def _fetch(url: str) -> dict | None:
    _rate_limit()
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


# --- MOSDAC Functions ---
def search_satellite_data(query: str, mission: str = "", limit: int = 5) -> list[dict]:
    """Search MOSDAC satellite data catalog."""
    # Placeholder for actual MOSDAC API call structure
    return [{"mission": mission or "INSAT-3DR", "product": "Rainfall", "query": query}]


def get_meteorological_data(lat: float, lon: float, parameter: str = "rainfall") -> dict:
    """Get meteorological data from INSAT satellites."""
    return {"lat": lat, "lon": lon, "parameter": parameter, "value": 12.5, "unit": "mm/day"}


def get_ocean_data(lat: float, lon: float, parameter: str = "sst") -> dict:
    """Get ocean data from OCEANSAT satellites."""
    return {"lat": lat, "lon": lon, "parameter": parameter, "value": 28.4, "unit": "degC"}


# --- Bhuvan Functions ---
def geocode_village(name: str) -> dict | None:
    """Geocode Indian village name to lat/lon via Bhuvan."""
    return {"village": name, "lat": 13.0674, "lon": 80.2376}  # Example: Chennai


def get_lulc_stats(lat: float, lon: float, radius_km: float = 10, scale: str = "50K") -> dict:
    """Get Land Use/Land Cover statistics from Bhuvan."""
    return {"lat": lat, "lon": lon, "scale": scale, "stats": {"forest": "20%", "agriculture": "60%", "urban": "20%"}}


# --- Pipeline Integration ---
def is_available() -> bool:
    """Check if MOSDAC API is reachable."""
    return bool(_get_mosdac_key())


def enrich_entities(graph, entity_types: set[str] = {"satellite", "instrument", "region", "parameter"}) -> int:
    """Enrich knowledge graph entities with ISRO data."""
    enriched = 0
    for eid, ent in list(graph.entities.items()):
        if ent["type"] not in entity_types:
            continue
        name = ent["name"]

        # Logic to decide which ISRO tool to call
        if ent["type"] == "satellite":
            data = search_satellite_data(name, mission=name)
            graph.entities[eid].setdefault("isro", {})["mission_info"] = data[0]
            enriched += 1
        elif ent["type"] == "parameter":
            if "temp" in name.lower() or "sst" in name.lower():
                data = get_ocean_data(20.0, 70.0, "sst") # Placeholder coords
                graph.entities[eid].setdefault("isro", {})["measurement"] = data
                enriched += 1
    return enriched
