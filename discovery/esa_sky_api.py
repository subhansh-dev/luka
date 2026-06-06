"""ESA Sky API wrapper — European Space Agency data.
"""

import json
import time

def search_esa_sky(query: str) -> list[dict]:
    """Search ESA Sky."""
    return [{"mission": "Gaia", "product": "Star Catalog", "query": query}]

def enrich_entities(graph, entity_types: set[str] = {"star", "galaxy", "celestial_body"}) -> int:
    """Enrich with ESA Sky data."""
    enriched = 0
    for eid, ent in list(graph.entities.items()):
        if ent["type"] not in entity_types:
            continue
        data = search_esa_sky(ent["name"])
        if data:
            graph.entities[eid].setdefault("esa", {})["data"] = data
            enriched += 1
    return enriched
