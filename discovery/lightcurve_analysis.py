"""Light curve analysis for exoplanet transit detection using lightkurve."""
import json
_HAS_LIGHTKURVE = False
try:
    import lightkurve as lk
    _HAS_LIGHTKURVE = True
except ImportError:
    pass

def search_lightcurves(target, mission="Kepler", sector=None, limit=5):
    """Search for light curves of a target star."""
    if not _HAS_LIGHTKURVE:
        return {"error": "lightkurve not available", "status": "missing_library"}
    try:
        search_result = lk.search_lightcurve(target, mission=mission)
        if len(search_result) == 0:
            return {"error": f"No light curves found for {target}", "status": "not_found"}
        results = []
        for lc in search_result[:limit]:
            results.append({
                "target": target, "mission": str(lc.mission),
                "author": str(lc.author), "exptime": float(lc.exptime.value) if hasattr(lc.exptime, "value") else lc.exptime,
                "t_start": float(lc.t_start.value) if hasattr(lc.t_start, "value") else lc.t_start,
                "t_end": float(lc.t_end.value) if hasattr(lc.t_end, "value") else lc.t_end,
            })
        return {"target": target, "count": len(results), "lightcurves": results, "status": "ok"}
    except Exception as e:
        return {"error": str(e)[:200], "status": "error"}

def detect_transits(target, mission="Kepler", period_days=None):
    """Detect transits in a light curve using BLS or given period."""
    if not _HAS_LIGHTKURVE:
        return {"error": "lightkurve not available", "status": "missing_library"}
    try:
        search_result = lk.search_lightcurve(target, mission=mission)
        if len(search_result) == 0:
            return {"error": f"No light curves for {target}", "status": "not_found"}
        lc = search_result[0].download()
        if lc is None:
            return {"error": "Download failed", "status": "download_error"}
        lc = lc.remove_nans().remove_outliers()
        if period_days:
            folded = lc.fold(period=period_days)
            depth = float(folded.flux.min().value) if hasattr(folded.flux.min(), "value") else float(folded.flux.min())
            return {"target": target, "period_days": period_days,
                    "transit_depth": depth, "n_points": len(folded), "status": "ok"}
        # BLS periodogram
        from lightkurve.periodogram import BoxLeastSquaresPeriodogram
        pg = lc.to_periodogram(method="bls", frequency_factor=1)
        best_period = float(pg.period_at_max_power.value)
        best_t0 = float(pg.transit_time_at_max_power.value)
        depth = float(pg.depth_at_max_power.value)
        return {"target": target, "best_period_days": round(best_period, 4),
                "transit_epoch": best_t0, "transit_depth": depth,
                "snr": float(pg.snr().max().value), "status": "ok"}
    except Exception as e:
        return {"error": str(e)[:200], "status": "error"}

def is_available():
    return _HAS_LIGHTKURVE
