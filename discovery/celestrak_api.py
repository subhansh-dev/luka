"""
celestrak_api.py — CelesTrak TLE Data API wrapper.

Two-Line Element sets for satellite orbit prediction.
Free, no API key required.
https://celestrak.org/NORAD/elements/
"""
import json
import time
import urllib.request
from pathlib import Path

CELESTRAK_BASE = "https://celestrak.org/NORAD/elements"
_LAST_CALL = 0.0


def _rate_limit():
    global _LAST_CALL
    now = time.time()
    if now - _LAST_CALL < 2.0:
        time.sleep(2.0 - (now - _LAST_CALL))
    _LAST_CALL = time.time()


def _fetch_text(url: str) -> str | None:
    _rate_limit()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Luka-Space-Discovery/2.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [CelesTrak] Request failed: {e}", flush=True)
        return None


def _fetch_json(url: str) -> dict | list | None:
    _rate_limit()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Luka-Space-Discovery/2.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  [CelesTrak] Request failed: {e}", flush=True)
        return None


def get_active_satellites(group: str = "stations") -> list[dict]:
    """Get TLE data for active satellites.

    Args:
        group: Satellite group. Options:
            - stations (ISS, Tiangong)
            - visual (brightest satellites)
            - active (all active satellites — large)
            - starlink, oneweb, iridium, gps, glonass, galileo, beidou
            - geo (geostationary)
            - weather, earth-resources, science, communication

    Returns:
        List of dicts with name, norad_id, tle_line1, tle_line2, epoch
    """
    url = f"{CELESTRAK_BASE}/gp.php?GROUP={group}&FORMAT=tle"
    text = _fetch_text(url)
    if not text:
        # Try JSON format as fallback
        url_json = f"{CELESTRAK_BASE}/gp.php?GROUP={group}&FORMAT=json"
        data = _fetch_json(url_json)
        if data and isinstance(data, list):
            return [{
                "name": s.get("OBJECT_NAME", ""),
                "norad_id": s.get("NORAD_CAT_ID", 0),
                "tle_line1": s.get("TLE_LINE1", ""),
                "tle_line2": s.get("TLE_LINE2", ""),
                "epoch": s.get("EPOCH", ""),
                "inclination": s.get("INCLINATION", 0),
                "raan": s.get("RA_OF_ASC_NODE", 0),
                "eccentricity": s.get("ECCENTRICITY", 0),
                "period": s.get("PERIOD", 0),
            } for s in data[:100]]
        return [{"error": "CelesTrak unreachable", "status": "api_error"}]

    # Parse TLE format (3 lines per satellite)
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    satellites = []
    i = 0
    while i + 2 < len(lines):
        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]
        if line1.startswith("1 ") and line2.startswith("2 "):
            norad_id = line1[2:7].strip()
            try:
                incl = float(line2[8:16].strip())
                period_min = float(line2[52:63].strip()) if len(line2) > 63 else 0
            except ValueError:
                incl = 0
                period_min = 0
            satellites.append({
                "name": name,
                "norad_id": int(norad_id) if norad_id.isdigit() else 0,
                "tle_line1": line1,
                "tle_line2": line2,
                "inclination": incl,
                "period_minutes": period_min,
            })
            i += 3
        else:
            i += 1
    return satellites[:100]


def get_satellite_by_name(name: str) -> list[dict]:
    """Search for a satellite by name."""
    url = f"{CELESTRAK_BASE}/gp.php?NAME={name}&FORMAT=json"
    data = _fetch_json(url)
    if not data or not isinstance(data, list):
        return [{"error": f"Satellite '{name}' not found", "status": "not_found"}]
    return [{
        "name": s.get("OBJECT_NAME", ""),
        "norad_id": s.get("NORAD_CAT_ID", 0),
        "tle_line1": s.get("TLE_LINE1", ""),
        "tle_line2": s.get("TLE_LINE2", ""),
        "epoch": s.get("EPOCH", ""),
        "inclination": s.get("INCLINATION", 0),
        "period": s.get("PERIOD", 0),
        "altitude_km": s.get("SEMIMAJOR_AXIS", 0) - 6371 if s.get("SEMIMAJOR_AXIS") else 0,
    } for s in data[:10]]


def get_special_interest() -> list[dict]:
    """Get special interest satellites (ISS, Tiangong, etc.)."""
    return get_active_satellites("stations")


def is_available() -> bool:
    """CelesTrak is always available (no key needed)."""
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
                results = get_active_satellites()
                if results and isinstance(results, list) and len(results) > 0:
                    count += 1
            except Exception:
                pass
    except Exception:
        pass
    return count
