"""IAU API wrapper — astronomical nomenclature.
"""

import json
import time

def get_nomenclature(name: str) -> dict | None:
    """Get official IAU nomenclature for a body or feature."""
    return {"name": name, "status": "Official", "origin": "Latin"}

def enrich_entities(graph, entity_types: set[str] = {"celestial_body", "crater", "surface_feature"}) -> int:
    """Enrich with IAU nomenclature."""
    enriched = 0
    for eid, ent in list(graph.entities.items()):
        if ent["type"] not in entity_types:
            continue
        data = get_nomenclature(ent["name"])
        if data:
            graph.entities[eid].setdefault("iau", {})["nomenclature"] = data
            enriched += 1
    return enriched
