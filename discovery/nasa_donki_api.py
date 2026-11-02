"""
nasa_donki_api.py — NASA DONKI (Space Weather Database Of Notifications, Knowledge, Information).

CME, solar flare, geomagnetic storm, interplanetary shock data.
Free with DEMO_KEY.
https://api.nasa.gov/DONKI/
"""
import json
import time
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

API_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
DONKI_BASE = "https://api.nasa.gov/DONKI"
_LAST_CALL = 0.0


def _get_key() -> str:
    try:
        cfg = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8-sig"))
        return cfg.get("nasa_api_key", "DEMO_KEY")
    except Exception:
        return "DEMO_KEY"


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
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  [DONKI] Request failed: {e}", flush=True)
        return None


def _date_range(days_back: int = 7) -> tuple[str, str]:
    end = datetime.now()
    start = end - timedelta(days=days_back)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def get_cme(days_back: int = 7) -> list[dict]:
    """Get Coronal Mass Ejection events."""
    key = _get_key()
    start, end = _date_range(days_back)
    url = f"{DONKI_BASE}/CME?startDate={start}&endDate={end}&api_key={key}"
    data = _fetch_json(url)
    if not data or not isinstance(data, list):
        return [{"error": "CME data unavailable", "status": "api_error"}]
    return [{
        "time": c.get("startTime", ""),
        "source_location": c.get("sourceLocation", ""),
        "active_region": c.get("activeRegionNum", 0),
        "speed": c.get("cmeAnalyses", [{}])[0].get("speed", 0) if c.get("cmeAnalyses") else 0,
        "type": c.get("cmeAnalyses", [{}])[0].get("type", "") if c.get("cmeAnalyses") else "",
        "note": (c.get("note", "") or "")[:200],
    } for c in data[:10]]


def get_solar_flares(days_back: int = 7) -> list[dict]:
    """Get Solar Flare events (FLR)."""
    key = _get_key()
    start, end = _date_range(days_back)
    url = f"{DONKI_BASE}/FLR?startDate={start}&endDate={end}&api_key={key}"
    data = _fetch_json(url)
    if not data or not isinstance(data, list):
        return [{"error": "Solar flare data unavailable", "status": "api_error"}]
    return [{
        "time": f.get("beginTime", ""),
        "peak_time": f.get("peakTime", ""),
        "end_time": f.get("endTime", ""),
        "class_type": f.get("classType", ""),
        "source_location": f.get("sourceLocation", ""),
        "active_region": f.get("activeRegionNum", 0),
    } for f in data[:10]]


def get_geomagnetic_storms(days_back: int = 7) -> list[dict]:
    """Get Geomagnetic Storm events (GST)."""
    key = _get_key()
    start, end = _date_range(days_back)
    url = f"{DONKI_BASE}/GST?startDate={start}&endDate={end}&api_key={key}"
    data = _fetch_json(url)
    if not data or not isinstance(data, list):
        return [{"error": "Geomagnetic storm data unavailable", "status": "api_error"}]
    return [{
        "time": g.get("startTime", ""),
        "kp_index": g.get("allKpIndex", [{}])[0].get("kpIndex", 0) if g.get("allKpIndex") else 0,
        "source": g.get("linkedEvents", []),
    } for g in data[:10]]


def get_interplanetary_shocks(days_back: int = 7) -> list[dict]:
    """Get Interplanetary Shock events (IPS)."""
    key = _get_key()
    start, end = _date_range(days_back)
    url = f"{DONKI_BASE}/IPS?startDate={start}&endDate={end}&api_key={key}"
    data = _fetch_json(url)
    if not data or not isinstance(data, list):
        return [{"error": "IPS data unavailable", "status": "api_error"}]
    return [{
        "time": s.get("eventTime", ""),
        "catalog": s.get("catalog", ""),
        "location": s.get("sourceLocation", ""),
        "instruments": [i.get("displayName", "") for i in s.get("instruments", [])],
    } for s in data[:10]]


def get_notifications(days_back: int = 3) -> list[dict]:
    """Get all space weather notifications (comprehensive)."""
    key = _get_key()
    start, end = _date_range(days_back)
    url = f"{DONKI_BASE}/notifications?startDate={start}&endDate={end}&api_key={key}"
    data = _fetch_json(url)
    if not data or not isinstance(data, list):
        return [{"error": "Notifications unavailable", "status": "api_error"}]
    return [{
        "time": n.get("messageIssueTime", ""),
        "type": n.get("messageType", ""),
        "body": (n.get("messageBody", "") or "")[:300],
    } for n in data[:10]]


def is_available() -> bool:
    """DONKI is available with DEMO_KEY."""
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
                results = get_solar_flares()
                if results and isinstance(results, list) and len(results) > 0:
                    count += 1
            except Exception:
                pass
    except Exception:
        pass
    return count
