"""SIMBAD API wrapper - upgraded with astroquery."""
import time
_HAS_ASTROQUERY = False
try:
    from astroquery.simbad import Simbad
    _HAS_ASTROQUERY = True
except ImportError:
    pass
_LAST = 0.0
def _rl():
    global _LAST; now=time.time()
    if now-_LAST<1: time.sleep(1-(now-_LAST))
    _LAST=time.time()
def search_object(name):
    _rl()
    if not _HAS_ASTROQUERY: return None
    try:
        result = Simbad.query_object(name)
        if result is None: return None
        rows = []
        for row in result:
            d = {}
            for col in result.colnames:
                val = row[col]
                d[col] = val.item() if hasattr(val, "item") else val
            rows.append(d)
        return rows[0] if rows else None
    except Exception as e:
        print(f"  [SIMBAD] {e}", flush=True)
        return None
def search_objects_in_region(ra, dec, radius_arcmin=10):
    _rl()
    if not _HAS_ASTROQUERY: return []
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs")
        result = Simbad.query_region(coord, radius=radius_arcmin * u.arcmin)
        if result is None: return []
        rows = []
        for row in result:
            d = {}
            for col in result.colnames:
                val = row[col]
                d[col] = val.item() if hasattr(val, "item") else val
            rows.append(d)
        return rows
    except Exception as e:
        print(f"  [SIMBAD] {e}", flush=True)
        return []
def is_available():
    return _HAS_ASTROQUERY
