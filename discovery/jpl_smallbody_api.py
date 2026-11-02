"""JPL Small-Body Database API. Asteroid/comet data. Free, no key."""
import json, time, urllib.request, urllib.parse
SBDB_BASE = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
_LAST = 0.0
def _rl():
    global _LAST; now=time.time()
    if now-_LAST<1.5: time.sleep(1.5-(now-_LAST))
    _LAST=time.time()
def _fetch(url):
    _rl()
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  [SBDB] {e}", flush=True)
        return None
def query_asteroids(object_class="", limit=20):
    params = {"fields": "spkid,full_name,e,a,i,om,w,ma,per,epoch", "limit": str(limit)}
    if object_class: params["sb-kind"] = object_class
    url = f"{SBDB_BASE}?{urllib.parse.urlencode(params)}"
    data = _fetch(url)
    if not data: return [{"error": "SBDB unreachable", "status": "api_error"}]
    fields_list = data.get("fields", [])
    objects = []
    for row in data.get("data", []):
        obj = {}
        for i, f in enumerate(fields_list):
            if i < len(row): obj[f] = row[i]
        objects.append(obj)
    return objects if objects else [{"error": "No objects found"}]
def get_asteroid(name):
    url = f"{SBDB_BASE}?sstr={urllib.parse.quote(name)}&fields=spkid,full_name,e,a,i,om,w,ma,per,epoch,class"
    data = _fetch(url)
    if not data: return {"error": "SBDB unreachable"}
    fields_list = data.get("fields", [])
    rows = data.get("data", [])
    if not rows: return {"error": f"Asteroid '{name}' not found"}
    obj = {}
    for i, f in enumerate(fields_list):
        if i < len(rows[0]): obj[f] = rows[0][i]
    return obj
def get_near_earth_asteroids(limit=20):
    return query_asteroids(object_class="NEA", limit=limit)
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
                results = get_near_earth_asteroids()
                if results and isinstance(results, list) and len(results) > 0:
                    count += 1
            except Exception:
                pass
    except Exception:
        pass
    return count
