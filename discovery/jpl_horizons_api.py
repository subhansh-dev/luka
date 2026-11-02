"""JPL Horizons API wrapper - upgraded with astroquery."""
import time
_HAS_ASTROQUERY = False
try:
    from astroquery.jplhorizons import Horizons
    _HAS_ASTROQUERY = True
except ImportError:
    pass
HORIZONS_IDS = {
    "mercury":"199","venus":"299","earth":"399","mars":"499",
    "jupiter":"599","saturn":"699","uranus":"799","neptune":"899",
    "pluto":"999","moon":"301","io":"501","europa":"502",
    "ganymede":"503","callisto":"504","titan":"606","enceladus":"602",
    "ceres":"1","vesta":"4","bennu":"2162173","iss":"-125544",
}
_LAST = 0.0
def _rl():
    global _LAST; now=time.time()
    if now-_LAST<1: time.sleep(1-(now-_LAST))
    _LAST=time.time()
def _resolve_id(target):
    t = target.strip().lower()
    if t in HORIZONS_IDS: return HORIZONS_IDS[t]
    return target.strip()
def get_ephemeris(target, epochs="now"):
    _rl()
    if not _HAS_ASTROQUERY: return {"target": target, "error": "astroquery not available"}
    try:
        import astropy.time
        target_id = _resolve_id(target)
        if epochs == "now":
            epochs = astropy.time.Time.now().jd
        obj = Horizons(id=target_id, location="500@399", epochs=epochs)
        eph = obj.ephemerides()
        if eph is None or len(eph) == 0:
            return {"target": target, "error": "No ephemeris"}
        row = eph[0]
        result = {"target": target, "source": "astroquery"}
        for col in eph.colnames:
            val = row[col]
            result[col] = val.item() if hasattr(val, "item") else val
        result["distance_au"] = row.get("delta", row.get("r", 0))
        result["velocity_km_s"] = row.get("rate", 0)
        result["magnitude"] = row.get("V", 0)
        return result
    except Exception as e:
        print(f"  [Horizons] {e}", flush=True)
        return {"target": target, "error": str(e)[:100]}
def get_orbital_elements(target):
    _rl()
    if not _HAS_ASTROQUERY: return {"error": "unavailable"}
    try:
        import astropy.time
        target_id = _resolve_id(target)
        jd = astropy.time.Time.now().jd
        obj = Horizons(id=target_id, location="500@399", epochs=jd)
        eph = obj.elements()
        if eph is None or len(eph) == 0: return {"error": f"No elements for {target}"}
        row = eph[0]
        result = {"target": target, "source": "astroquery"}
        for col in eph.colnames:
            val = row[col]
            result[col] = val.item() if hasattr(val, "item") else val
        return result
    except Exception as e:
        print(f"  [Horizons] {e}", flush=True)
        return {"error": str(e)[:100]}
def is_available():
    return _HAS_ASTROQUERY
