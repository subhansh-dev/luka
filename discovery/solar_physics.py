"""Solar physics analysis using sunpy."""
import json
_HAS_SUNPY = False
try:
    import sunpy.map
    import sunpy.timeseries
    from sunpy.net import Fido, attrs as a
    _HAS_SUNPY = True
except ImportError:
    pass

def search_solar_data(instrument="AIA", wavelength=171, start_time="2024-01-01", end_time="2024-01-02"):
    """Search for solar observation data."""
    if not _HAS_SUNPY:
        return {"error": "sunpy not available", "status": "missing_library"}
    try:
        result = Fido.search(a.Time(start_time, end_time), a.Instrument(instrument), a.Wavelength(wavelength * u.AA))
        n_results = len(result)
        files = []
        for table in result:
            for row in table[:5]:
                files.append({"instrument": instrument, "wavelength": wavelength,
                              "time": str(row.get("Start Time", "")),
                              "url": str(row.get("Url", ""))})
        return {"count": n_results, "files": files[:10], "status": "ok"}
    except Exception as e:
        return {"error": str(e)[:200], "status": "error"}

def get_solar_indices():
    """Get current solar activity indices."""
    if not _HAS_SUNPY:
        return {"error": "sunpy not available"}
    try:
        result = Fido.search(a.Time("2024-01-01", "2024-01-02"), a.Instrument("GOES"))
        return {"count": len(result), "status": "ok"}
    except Exception as e:
        return {"error": str(e)[:200]}

def is_available():
    return _HAS_SUNPY
