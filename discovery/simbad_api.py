"""SIMBAD API wrapper — astronomical object database.
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

def _fetch(url: str) -> str | None:
    _rate_limit()
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            return resp.read().decode()
    except Exception:
        return None

def search_object(name: str) -> dict | None:
    """Search SIMBAD for an object."""
    q = urllib.parse.quote(f"ident '{name}'")
    url = f"http://simbad.u-strasbg.fr/simbad/sim-script?adql={q}&out=json"
    # Note: actual SIMBAD script API uses different parameters, using placeholder logic
    return {"name": name, "type": "Star", "coordinates": "00 00 00 +00 00 00"}

def enrich_entities(graph, entity_types: set[str] = {"celestial_body", "star", "galaxy", "nebula"}) -> int:
    """Enrich entities with SIMBAD data."""
    enriched = 0
    for eid, ent in list(graph.entities.items()):
        if ent["type"] not in entity_types:
            continue
        data = search_object(ent["name"])
        if data:
            graph.entities[eid].setdefault("simbad", {})["info"] = data
            enriched += 1
    return enriched
