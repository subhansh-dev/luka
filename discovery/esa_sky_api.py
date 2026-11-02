"""ESA Sky API wrapper - upgraded with astroquery."""
import time
_HAS_ASTROQUERY = False
try:
    from astroquery.esasky import ESASky
    _HAS_ASTROQUERY = True
except ImportError:
    pass
_LAST = 0.0
def _rl():
    global _LAST; now=time.time()
    if now-_LAST<2: time.sleep(2-(now-_LAST))
    _LAST=time.time()
def search_esa_sky(target_name):
    _rl()
    if not _HAS_ASTROQUERY: return []
    try:
        result = ESASky.query_object_catalogs(target_name)
        if not result: return []
        all_results = []
        for catalog_name, table in result.items():
            for row in table[:10]:
                d = {"catalog": catalog_name}
                for col in table.colnames:
                    val = row[col]
                    d[col] = val.item() if hasattr(val, "item") else val
                all_results.append(d)
        return all_results
    except Exception as e:
        print(f"  [ESASky] {e}", flush=True)
        return []
def is_available():
    return _HAS_ASTROQUERY
