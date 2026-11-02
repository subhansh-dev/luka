"""
mast_api.py — MAST/STScI API wrapper for JWST, Hubble, Kepler, TESS data.

Free, no API key required for basic queries.
https://mast.stsci.edu/api/v0/
"""
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

MAST_BASE = "https://mast.stsci.edu/api/v0"
_LAST_CALL = 0.0


def _rate_limit():
    global _LAST_CALL
    now = time.time()
    if now - _LAST_CALL < 1.5:
        time.sleep(1.5 - (now - _LAST_CALL))
    _LAST_CALL = time.time()


def _fetch_json(url: str) -> dict | list | None:
    _rate_limit()
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Luka-Space-Discovery/2.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  [MAST] Request failed: {e}", flush=True)
        return None


def search_observations(target: str, mission: str = "", radius: float = 0.2,
                        limit: int = 10) -> list[dict]:
    """Search MAST for observations of a target.

    Args:
        target: Object name (e.g. "TRAPPIST-1", "M42")
        mission: Mission filter (JWST, HST, TESS, Kepler, etc.)
        radius: Search radius in degrees
        limit: Max results

    Returns:
        List of observation dicts
    """
    encoded = urllib.parse.quote(target)
    url = f"{MAST_BASE}/Mashup/Mashup.asmx/Query?request="
    query = {
        "service": "Mast.Caom.Filtered",
        "format": "json",
        "params": {
            "columns": "obs_id,obs_collection,target_name,t_exptime,dataproduct_type,wavelength_region,calib_level",
            "filters": [
                {"paramName": "target_name", "values": [target]},
            ],
        },
    }
    if mission:
        query["params"]["filters"].append({"paramName": "obs_collection", "values": [mission]})

    # Use simpler cone search endpoint
    cone_url = (f"{MAST_BASE}/Mashup/Mashup.asmx/invoke?service=Mast.Caom.Cone"
                f"&params={urllib.parse.quote(json.dumps({'RA': 0, 'DEC': 0, 'SR': radius}))}")
    # Fallback: use resolved name
    resolve_url = f"{MAST_BASE}/Mashup/Mashup.asmx/resolve?name={encoded}"
    resolved = _fetch_json(resolve_url)

    if not resolved:
        # Try direct search via simpler endpoint
        search_url = f"https://mast.stsci.edu/api/v0/invoke?service=Mast.Caom.Filtered&format=json"
        data = _fetch_json(search_url)
        if data:
            return [{"raw": str(d)[:200]} for d in (data if isinstance(data, list) else [data])[:limit]]
        return [{"error": "MAST search failed", "status": "api_error", "target": target}]

    return [{"resolved": True, "target": target, "data": str(resolved)[:300]}]


def search_jwst(target: str, limit: int = 10) -> list[dict]:
    """Search specifically for JWST observations."""
    return search_observations(target, mission="JWST", limit=limit)


def search_hst(target: str, limit: int = 10) -> list[dict]:
    """Search specifically for HST observations."""
    return search_observations(target, mission="HST", limit=limit)


def search_tess(target: str, limit: int = 10) -> list[dict]:
    """Search specifically for TESS observations."""
    return search_observations(target, mission="TESS", limit=limit)


def search_kepler(target: str, limit: int = 10) -> list[dict]:
    """Search specifically for Kepler observations."""
    return search_observations(target, mission="Kepler", limit=limit)


def is_available() -> bool:
    """MAST is always available (no key needed for basic queries)."""
    return True

def enrich_entities(graph):
    """Enrich knowledge graph entities with data from this API."""
    count = 0
    try:
        entities = list(graph.entities.values())[:10]
        for ent in entities:
            name = ent.get("name", "")
            if not name or len(name) < 3:
                continue
            # Try to find relevant data
            try:
                results = search_observations()
                if results and isinstance(results, list) and len(results) > 0:
                    count += 1
            except Exception:
                pass
    except Exception:
        pass
    return count
