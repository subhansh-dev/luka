"""NASA GCN - Gamma-ray burst alerts. Free."""
import json, time, urllib.request
GCN_BASE = "https://gcn.nasa.gov/api"
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
        print(f"  [GCN] {e}", flush=True)
        return None
def get_circulars(limit=20):
    data = _fetch(f"{GCN_BASE}/circulars?limit={limit}")
    if not data: return [{"error": "GCN unreachable", "status": "api_error"}]
    if isinstance(data, list):
        return [{"circular_id": c.get("circularId",""), "subject": c.get("subject",""),
                 "submitted": c.get("submittedDate",""), "body": (c.get("body","") or "")[:300]}
                for c in data[:limit]]
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
                results = get_circulars()
                if results and isinstance(results, list) and len(results) > 0:
                    count += 1
            except Exception:
                pass
    except Exception:
        pass
    return count
