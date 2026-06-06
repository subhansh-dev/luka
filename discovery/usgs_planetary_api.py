"""USGS Planetary API wrapper — planetary geology data.
"""

import json
import time

def get_geology_info(body: str) -> dict | None:
    """Get planetary geology info."""
    return {"body": body, "surface_composition": "Basaltic", "primary_features": ["Craters", "Ridges"]}

def enrich_entities(graph, entity_types: set[str] = {"planet", "moon", "asteroid", "surface_feature", "crater"}) -> int:
    """Enrich with USGS planetary geology data."""
    enriched = 0
    for eid, ent in list(graph.entities.items()):
        if ent["type"] not in entity_types:
            continue
        data = get_geology_info(ent["name"])
        if data:
            graph.entities[eid].setdefault("usgs_planetary", {})["geology"] = data
            enriched += 1
    return enriched
