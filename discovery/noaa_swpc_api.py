"""
noaa_swpc_api.py — NOAA Space Weather Prediction Center API wrapper.

Solar wind, geomagnetic indices, CME data, solar flares.
Free, no API key required.
https://services.swpc.noaa.gov/
"""
import json
import time
import urllib.request
from pathlib import Path

NOAA_BASE = "https://services.swpc.noaa.gov"
_LAST_CALL = 0.0


def _rate_limit():
    global _LAST_CALL
    now = time.time()
    if now - _LAST_CALL < 1.0:
        time.sleep(1.0 - (now - _LAST_CALL))
    _LAST_CALL = time.time()


def _fetch_json(url: str) -> dict | list | None:
    _rate_limit()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Luka-Space-Discovery/2.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  [NOAA SWPC] Request failed: {e}", flush=True)
        return None


def get_solar_wind() -> dict | None:
    """Get real-time solar wind data (speed, density, temperature)."""
    data = _fetch_json(f"{NOAA_BASE}/products/solar-wind/plasma-7-day.json")
    if not data or not isinstance(data, list) or len(data) < 2:
        return {"error": "Solar wind data unavailable", "status": "api_error"}
    # First row is header, last row is most recent
    header = data[0]
    latest = data[-1]
    return {
        "time_tag": latest[0] if len(latest) > 0 else "",
        "density": latest[1] if len(latest) > 1 else None,
        "speed": latest[2] if len(latest) > 2 else None,
        "temperature": latest[3] if len(latest) > 3 else None,
        "status": "ok",
    }


def get_geomagnetic_indices() -> dict | None:
    """Get current geomagnetic indices (Kp, ap)."""
    data = _fetch_json(f"{NOAA_BASE}/products/noaa-planetary-k-index.json")
    if not data or not isinstance(data, list) or len(data) < 2:
        return {"error": "Geomagnetic data unavailable", "status": "api_error"}
    header = data[0]
    latest = data[-1]
    return {
        "time_tag": latest[0] if len(latest) > 0 else "",
        "kp_index": latest[1] if len(latest) > 1 else None,
        "estimated_kp": latest[2] if len(latest) > 2 else None,
        "kp_color": latest[3] if len(latest) > 3 else "",
        "status": "ok",
    }


def get_solar_flares() -> list[dict]:
    """Get recent solar flare events."""
    data = _fetch_json(f"{NOAA_BASE}/products/solar-cycle/overview.json")
    if not data or not isinstance(data, list):
        return [{"error": "Solar flare data unavailable", "status": "api_error"}]
    header = data[0] if data else []
    flares = []
    for row in data[-10:]:  # last 10 entries
        if len(row) >= 4:
            flares.append({
                "time_tag": row[0],
                "flux": row[1],
                "scale": row[2],
                "location": row[3] if len(row) > 3 else "",
            })
    return flares


def get_cme_alerts() -> list[dict]:
    """Get recent Coronal Mass Ejection alerts."""
    data = _fetch_json(f"{NOAA_BASE}/products/alerts.json")
    if not data or not isinstance(data, list):
        return [{"error": "CME alerts unavailable", "status": "api_error"}]
    cme_alerts = []
    for alert in data[:20]:
        if isinstance(alert, dict):
            msg = alert.get("message", "")
            if "CME" in msg or "coronal" in msg.lower():
                cme_alerts.append({
                    "time_tag": alert.get("issue_datetime", ""),
                    "message": msg[:200],
                    "alert_type": alert.get("alert_type", ""),
                })
    return cme_alerts[:10]


def get_3_day_forecast() -> dict | None:
    """Get 3-day space weather forecast."""
    data = _fetch_json(f"{NOAA_BASE}/text/3-day-forecast.json")
    if not data:
        return {"error": "Forecast unavailable", "status": "api_error"}
    return data if isinstance(data, dict) else {"forecast": str(data)[:500]}


def is_available() -> bool:
    """NOAA SWPC is always available (no key needed)."""
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
                results = get_solar_wind()
                if results and isinstance(results, list) and len(results) > 0:
                    count += 1
            except Exception:
                pass
    except Exception:
        pass
    return count
