"""VizieR API wrapper - upgraded with astroquery."""
import time
_HAS_ASTROQUERY = False
try:
    from astroquery.vizier import Vizier
    _HAS_ASTROQUERY = True
except ImportError:
    pass
_LAST = 0.0
def _rl():
    global _LAST; now=time.time()
    if now-_LAST<1: time.sleep(1-(now-_LAST))
    _LAST=time.time()
def search_catalogs(query, catalog=None, limit=20):
    _rl()
    if not _HAS_ASTROQUERY: return []
    try:
        v = Vizier(row_limit=limit)
        if catalog:
            result = v.query_object(query, catalog=catalog)
        else:
            result = v.query_object(query)
        if not result: return []
        all_rows = []
        for table in result:
            for row in table[:limit]:
                d = {}
                for col in table.colnames:
                    val = row[col]
                    d[col] = val.item() if hasattr(val, "item") else val
                d["_catalog"] = table.meta.get("name", "unknown")
                all_rows.append(d)
        return all_rows
    except Exception as e:
        print(f"  [VizieR] {e}", flush=True)
        return []
def search_object_in_gaia(object_name, limit=10):
    return search_catalogs(object_name, catalog="I/355/gaiadr3", limit=limit)
def is_available():
    return _HAS_ASTROQUERY
