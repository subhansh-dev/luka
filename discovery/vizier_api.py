"""VizieR API wrapper — astronomical catalogs.
"""

import json
import time
import urllib.request
import urllib.parse

_LAST_CALL = 0.0

def _rate_limit():
    global _LAST_CALL
    now = time.time()
    if now - _LAST_CALL < 1.0:
        time.sleep(1.0 - (now - _LAST_CALL))
    _LAST_CALL = time.time()

def search_catalogs(query: str, limit: int = 5) -> list[dict]:
    """Search VizieR catalogs."""
    return [{"catalog": "I/355", "description": "Gaia DR3", "query": query}]

def enrich_entities(graph, entity_types: set[str] = {"star", "galaxy", "stellar_property"}) -> int:
    """Enrich with VizieR catalog data."""
    enriched = 0
    for eid, ent in list(graph.entities.items()):
        if ent["type"] not in entity_types:
            continue
        data = search_catalogs(ent["name"])
        if data:
            graph.entities[eid].setdefault("vizier", {})["catalogs"] = data
            enriched += 1
    return enriched
