"""
n2yo_api.py — N2YO Satellite Tracking API wrapper.

Real-time satellite positions and passes.
Free tier: 1000 requests/day at n2yo.com/api/
"""
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

API_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
N2YO_BASE = "https://api.n2yo.com/rest/v1/satellite"
_LAST_CALL = 0.0


def _get_key() -> str:
    try:
        cfg = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8-sig"))
        return cfg.get("n2yo_api_key", "")
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
        req = urllib.request.Request(url, headers={"User-Agent": "Luka-Space-Discovery/2.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  [N2YO] Request failed: {e}", flush=True)
        return None


def get_satellite_positions(norad_id: int, observer_lat: float = 28.6,
                            observer_lng: float = 77.2, observer_alt: float = 0,
                            seconds: int = 1) -> list[dict]:
    """Get real-time satellite position.

    Args:
        norad_id: NORAD catalog ID (e.g., 25544 for ISS)
        observer_lat: Observer latitude (default: Delhi)
        observer_lng: Observer longitude
        observer_alt: Observer altitude in km
        seconds: Number of seconds of position data

    Returns:
        List of position dicts with lat, lng, altitude, azimuth, elevation
    """
    key = _get_key()
    if not key:
        return [{"error": "N2YO API key not configured", "status": "missing_key"}]
    url = (f"{N2YO_BASE}/positions/{norad_id}/{observer_lat}/{observer_lng}/{observer_alt}/{seconds}/&apiKey={key}")
    data = _fetch(url)
    if not data:
        return [{"error": "N2YO API unreachable", "status": "api_error"}]
    positions = data.get("positions", [])
    return [{
        "satname": data.get("info", {}).get("satname", ""),
        "satid": norad_id,
        "lat": p.get("satlatitude", 0),
        "lng": p.get("satlongitude", 0),
        "altitude": p.get("sataltitude", 0),
        "azimuth": p.get("azimuth", 0),
        "elevation": p.get("elevation", 0),
        "timestamp": p.get("timestamp", 0),
    } for p in positions]


def get_visual_passes(norad_id: int, observer_lat: float = 28.6,
                      observer_lng: float = 77.2, observer_alt: float = 0,
                      days: int = 2, min_elevation: int = 10) -> list[dict]:
    """Get upcoming visual passes of a satellite.

    Args:
        norad_id: NORAD catalog ID
        observer_lat/lng/alt: Observer location
        days: Number of days to look ahead
        min_elevation: Minimum elevation in degrees

    Returns:
        List of pass dicts with start/end times, duration, max elevation
    """
    key = _get_key()
    if not key:
        return [{"error": "N2YO API key not configured", "status": "missing_key"}]
    url = (f"{N2YO_BASE}/visualpasses/{norad_id}/{observer_lat}/{observer_lng}/{observer_alt}/{days}/{min_elevation}/&apiKey={key}")
    data = _fetch(url)
    if not data:
        return [{"error": "N2YO API unreachable", "status": "api_error"}]
    passes = data.get("passes", [])
    return [{
        "satname": data.get("info", {}).get("satname", ""),
        "start": p.get("startUTC", 0),
        "max": p.get("maxUTC", 0),
        "end": p.get("endUTC", 0),
        "duration": p.get("duration", 0),
        "max_elevation": p.get("maxEl", 0),
        "start_az": p.get("startAz", 0),
        "end_az": p.get("endAz", 0),
    } for p in passes]


def get_radio_passes(norad_id: int, observer_lat: float = 28.6,
                     observer_lng: float = 77.2, observer_alt: float = 0,
                     days: int = 2) -> list[dict]:
    """Get upcoming radio passes (for satellite communication)."""
    key = _get_key()
    if not key:
        return [{"error": "N2YO API key not configured"}]
    url = (f"{N2YO_BASE}/radiopasses/{norad_id}/{observer_lat}/{observer_lng}/{observer_alt}/{days}/&apiKey={key}")
    data = _fetch(url)
    if not data:
        return [{"error": "N2YO API unreachable"}]
    passes = data.get("passes", [])
    return [{
        "satname": data.get("info", {}).get("satname", ""),
        "start": p.get("startUTC", 0),
        "max": p.get("maxUTC", 0),
        "end": p.get("endUTC", 0),
        "duration": p.get("duration", 0),
        "max_elevation": p.get("maxEl", 0),
    } for p in passes]


def is_available() -> bool:
    """Check if N2YO API key is configured."""
    return bool(_get_key())
