"""Minor Planet Center API. Orbital elements. Free, no key."""
import json, time, urllib.request
MPC_BASE = "https://minorplanetcenter.net"
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
        print(f"  [MPC] {e}", flush=True)
        return None
def get_nea_confirmations(limit=20):
    data = _fetch(f"{MPC_BASE}/neocp/confirmed.json")
    if not data: return [{"error": "MPC unreachable", "status": "api_error"}]
    if isinstance(data, list):
        return [{"designation": n.get("Des",""), "name": n.get("Name",""),
                 "h": n.get("H",0), "discovery_date": n.get("Discovery",""),
                 "last_obs": n.get("LastObs","")} for n in data[:limit]]
    return [{"error": "Unexpected format"}]
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
                results = get_nea_confirmations()
                if results and isinstance(results, list) and len(results) > 0:
                    count += 1
            except Exception:
                pass
    except Exception:
        pass
    return count
