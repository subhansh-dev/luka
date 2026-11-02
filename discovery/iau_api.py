"""IAU API wrapper — International Astronomical Union nomenclature.

Uses the IAU Planetary Nomenclature database and Gazetteer of Planetary Nomenclature.
https://planetarynames.wr.usgs.gov/
"""

import json
import time
import urllib.request
import urllib.parse

GPN_BASE = "https://planetarynames.wr.usgs.gov"
_LAST_CALL = 0.0


def _rate_limit():
    global _LAST_CALL
    now = time.time()
    if now - _LAST_CALL < 1.0:
        time.sleep(1.0 - (now - _LAST_CALL))
    _LAST_CALL = time.time()


def _fetch_json(url: str) -> dict | None:
    _rate_limit()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Luka-Space-Discovery/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


# Known IAU-approved planetary feature categories
FEATURE_CATEGORIES = {
    "crater": "Impact crater",
    "mons": "Mountain",
    "planitia": "Plain",
    "tholus": "Small dome mountain",
    "vallis": "Valley",
    "patera": "Irregular crater",
    "chasma": "Deep, elongate, steep-sided depression",
    "regio": "Large area marked by distinctive albedo or color",
    "terra": "Extensive land mass",
    "fossa": "Long, narrow, shallow depression",
    "catena": "Chain of craters",
    "rupes": "Scarp",
    "dorsum": "Ridge",
    "labyrinthus": "Complex of intersecting valleys",
    "macula": "Dark spot",
    "unda": "Wave-form feature",
}


def get_nomenclature(name: str) -> dict | None:
    """Get IAU nomenclature information for an astronomical feature.

    Args:
        name: Feature name (e.g., "Olympus Mons", "Shackleton", "Tycho")

    Returns:
        Dict with name, status, feature_type, target_body, or None on failure.
    """
    safe_name = urllib.parse.quote(name.strip())

    # Try the GPN search endpoint
    url = f"{GPN_BASE}/api/Feature?feature={safe_name}"
    data = _fetch_json(url)

    if data and isinstance(data, list) and len(data) > 0:
        feature = data[0]
        return {
            "name": feature.get("name", name),
            "status": "IAU Approved" if feature.get("approvalStatus") == "Approved" else "Unknown",
            "feature_type": feature.get("featureType", "Unknown"),
            "target_body": feature.get("target", "Unknown"),
            "diameter_km": feature.get("diameter"),
            "latitude": feature.get("centerLatitude"),
            "longitude": feature.get("centerLongitude"),
            "origin": feature.get("origin", "Unknown"),
            "approval_year": feature.get("approvalDate"),
            "source": "IAU Gazetteer of Planetary Nomenclature",
        }

    # Fallback: check known feature type from name suffix
    name_lower = name.lower()
    for suffix, feature_type in FEATURE_CATEGORIES.items():
        if suffix in name_lower:
            return {
                "name": name,
                "status": "Inferred from nomenclature conventions",
                "feature_type": feature_type,
                "target_body": "Unknown",
                "source": "IAU naming conventions (inferred)",
            }

    # Last resort: return basic info indicating we searched but found nothing
    return {
        "name": name,
        "status": "Not found in IAU database",
        "feature_type": "Unknown",
        "source": "IAU Gazetteer (no match)",
    }


def search_features(target_body: str, feature_type: str = "",
                     limit: int = 20) -> list[dict]:
    """Search for IAU-approved features on a specific body.

    Args:
        target_body: Planetary body (e.g., "Mars", "Moon", "Europa")
        feature_type: Optional feature type filter
        limit: Max results

    Returns:
        List of feature dicts.
    """
    params = {"target": target_body, "limit": limit}
    if feature_type:
        params["featureType"] = feature_type

    url = f"{GPN_BASE}/api/Feature?{urllib.parse.urlencode(params)}"
    data = _fetch_json(url)

    if not data or not isinstance(data, list):
        return []

    results = []
    for feature in data[:limit]:
        results.append({
            "name": feature.get("name", ""),
            "feature_type": feature.get("featureType", ""),
            "target_body": feature.get("target", target_body),
            "diameter_km": feature.get("diameter"),
            "latitude": feature.get("centerLatitude"),
            "longitude": feature.get("centerLongitude"),
            "origin": feature.get("origin", ""),
            "status": feature.get("approvalStatus", ""),
        })
    return results


def enrich_entities(graph, entity_types: set[str] = {"celestial_body", "crater", "surface_feature"}) -> int:
    """Enrich entities with IAU nomenclature data."""
    enriched = 0
    for eid, ent in list(graph.entities.items()):
        if ent.get("type") not in entity_types:
            continue
        name = ent.get("name", "")
        if not name:
            continue
        data = get_nomenclature(name)
        if data:
            graph.entities[eid].setdefault("iau", {})["nomenclature"] = data
            enriched += 1
    return enriched
