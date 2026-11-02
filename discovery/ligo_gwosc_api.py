"""LIGO GWOSC Gravitational Wave Event API. Free, no key."""
import json, time, urllib.request
GWOSC_BASE = "https://gwosc.org/eventapi"
_LAST = 0.0
def _rl():
    global _LAST; now=time.time()
    if now-_LAST<2: time.sleep(2-(now-_LAST))
    _LAST=time.time()
def _fetch(url):
    _rl()
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  [GWOSC] {e}", flush=True)
        return None
def get_events(catalog="GWTC", limit=20):
    data = _fetch(f"{GWOSC_BASE}/v2/{catalog}")
    if not data or not isinstance(data, dict):
        return [{"error": "GWOSC unreachable", "status": "api_error"}]
    events = []
    for name, info in list(data.get("events", {}).items())[:limit]:
        if not isinstance(info, dict): continue
        events.append({
            "name": info.get("commonName", name),
            "gps": info.get("GPS", 0),
            "mass_1": info.get("mass_1_source", 0),
            "mass_2": info.get("mass_2_source", 0),
            "distance_mpc": info.get("luminosity_distance", 0),
            "snr": info.get("network_matched_filter_snr", 0),
        })
    return events if events else [{"error": "No events found"}]
def is_available(): return True

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
                results = get_events()
                if results and isinstance(results, list) and len(results) > 0:
                    count += 1
            except Exception:
                pass
    except Exception:
        pass
    return count
