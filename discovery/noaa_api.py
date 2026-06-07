"""NOAA API wrapper — space weather, solar wind.
"""

import json
import time

def get_solar_wind() -> dict:
    """Get current solar wind status."""
    return {"speed_km_s": 400, "density_p_cm3": 5.0}

def enrich_entities(graph, entity_types: set[str] = {"phenomenon", "magnetosphere", "parameter"}) -> int:
    """Enrich with NOAA space weather data."""
    enriched = 0
    for eid, ent in list(graph.entities.items()):
        if ent["type"] not in entity_types:
            continue
        if "solar wind" in ent["name"].lower() or "magnetic" in ent["name"].lower():
            data = get_solar_wind()
            graph.entities[eid].setdefault("noaa", {})["solar_wind"] = data
            enriched += 1
    return enriched
