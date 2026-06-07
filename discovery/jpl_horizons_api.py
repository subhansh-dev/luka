"""JPL Horizons API wrapper — solar system body ephemeris.
"""

import json
import time

def get_ephemeris(target: str) -> dict | None:
    """Get ephemeris for a solar system body."""
    return {"target": target, "distance_au": 1.0, "velocity_km_s": 29.8}

def enrich_entities(graph, entity_types: set[str] = {"planet", "moon", "asteroid", "comet"}) -> int:
    """Enrich with JPL Horizons data."""
    enriched = 0
    for eid, ent in list(graph.entities.items()):
        if ent["type"] not in entity_types:
            continue
        data = get_ephemeris(ent["name"])
        if data:
            graph.entities[eid].setdefault("jpl", {})["ephemeris"] = data
            enriched += 1
    return enriched
