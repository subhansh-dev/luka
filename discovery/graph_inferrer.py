"""graph_inferrer.py - Cross-entity relationship inference.

Connects entities that co-occur in papers but aren't directly connected.
Reduces knowledge graph fragmentation by bridging isolated clusters.
"""
import re
from collections import Counter

# Type-to-type relationship inference rules
TYPE_RELATIONSHIPS = {
    ("star", "exoplanet"): "hosts",
    ("star", "planet"): "hosts",
    ("star", "asteroid"): "orbits_around",
    ("celestial_body", "mission"): "observed_by",
    ("telescope", "exoplanet"): "observed",
    ("telescope", "star"): "observed",
    ("instrument", "celestial_body"): "measures",
    ("phenomenon", "celestial_body"): "affects",
    ("phenomenon", "star"): "occurs_in",
    ("phenomenon", "exoplanet"): "occurs_in",
    ("theory", "phenomenon"): "explains",
    ("theory", "celestial_body"): "describes",
    ("concept", "phenomenon"): "related_to",
    ("concept", "theory"): "supports",
    ("molecule_gas", "exoplanet"): "detected_in",
    ("molecule_gas", "star"): "found_in",
    ("spectral_feature", "molecule_gas"): "signature_of",
    ("atmospheric_param", "exoplanet"): "measured_for",
    ("stellar_property", "star"): "characterizes",
    ("orbit", "celestial_body"): "describes",
    ("spacecraft", "celestial_body"): "visits",
    ("spacecraft", "mission"): "part_of",
    ("galaxy", "star"): "contains",
    ("nebula", "star"): "birthplace_of",
    ("comet", "star"): "orbits_around",
    ("asteroid", "planet"): "near",
}

def infer_relationships(graph, min_cooccurrence=2):
    """Infer relationships between entities that co-occur in papers.

    Args:
        graph: KnowledgeGraph instance
        min_cooccurrence: Minimum number of shared papers to infer a relationship

    Returns:
        Number of new relationships added
    """
    if not graph or not graph.entities or not graph.papers:
        return 0

    # Build paper -> entities map
    paper_entities = {}
    for eid, ent in graph.entities.items():
        for pid in ent.get("papers", []):
            if pid not in paper_entities:
                paper_entities[pid] = []
            paper_entities[pid].append(eid)

    # Count co-occurrences
    cooccurrence = Counter()
    for pid, entities in paper_entities.items():
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                key = tuple(sorted([entities[i], entities[j]]))
                cooccurrence[key] += 1

    # Existing relationships (to avoid duplicates)
    existing = set()
    for rel in graph.relationships:
        key = tuple(sorted([rel.get("source", ""), rel.get("target", "")]))
        existing.add(key)

    # Infer new relationships
    added = 0
    for (eid1, eid2), count in cooccurrence.items():
        if count < min_cooccurrence:
            continue
        key = tuple(sorted([eid1, eid2]))
        if key in existing:
            continue

        ent1 = graph.entities.get(eid1, {})
        ent2 = graph.entities.get(eid2, {})
        type1 = ent1.get("type", "unknown")
        type2 = ent2.get("type", "unknown")

        # Find the best relationship type
        rel_type = _infer_relation_type(type1, type2)

        # Confidence based on co-occurrence count
        confidence = min(0.7, 0.4 + count * 0.05)

        graph.relationships.append({
            "source": eid1,
            "relation": rel_type,
            "target": eid2,
            "confidence": confidence,
            "papers": [],
            "inferred": True,
            "cooccurrence_count": count,
        })
        existing.add(key)
        added += 1

    return added


def _infer_relation_type(type1, type2):
    """Find the best relationship type for two entity types."""
    # Check both orderings
    pair = (type1, type2)
    if pair in TYPE_RELATIONSHIPS:
        return TYPE_RELATIONSHIPS[pair]
    pair_rev = (type2, type1)
    if pair_rev in TYPE_RELATIONSHIPS:
        return TYPE_RELATIONSHIPS[pair_rev]

    # Fallback based on type categories
    celestial = {"celestial_body", "star", "exoplanet", "planet", "asteroid",
                 "comet", "galaxy", "nebula", "moon"}
    instruments = {"telescope", "instrument", "spacecraft"}
    concepts = {"theory", "concept", "phenomenon", "molecule_gas",
                "spectral_feature", "atmospheric_param", "stellar_property", "orbit"}

    if type1 in celestial and type2 in celestial:
        return "associated_with"
    if type1 in instruments and type2 in celestial:
        return "observes"
    if type1 in concepts and type2 in celestial:
        return "characterizes"
    if type1 in concepts and type2 in concepts:
        return "related_to"

    return "co_occurs_with"


def reduce_fragmentation(graph, min_cooccurrence=2):
    """Full fragmentation reduction pass.

    1. Infer cross-entity relationships
    2. Count connected components before/after
    3. Return stats
    """
    components_before = _count_components(graph)

    added = infer_relationships(graph, min_cooccurrence)

    components_after = _count_components(graph)

    return {
        "relationships_added": added,
        "components_before": components_before,
        "components_after": components_after,
        "components_reduced": components_before - components_after,
    }


def _count_components(graph):
    """Count connected components in the graph using union-find."""
    if not graph or not graph.entities:
        return 0

    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for rel in graph.relationships:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        if src and tgt:
            union(src, tgt)

    # Count unique roots
    roots = set()
    for eid in graph.entities:
        roots.add(find(eid))

    return len(roots)


def is_available():
    return True
