#!/usr/bin/env python3
"""
creativity_engine.py — Luka Computational Creativity Engine (Fauconnier & Turner)
===================================================================================

Research-grade computational creativity implementing Fauconnier & Turner's
Conceptual Blending Theory (1998, 2002) and Koestler's Bisociation (1964).

Core Architecture:
  ┌──────────────────────────────────────────────────────────────────────┐
  │                   CONCEPTUAL INTEGRATION NETWORK                     │
  ├──────────────────────────────────────────────────────────────────────┤
  │  Input 1 ──────────┐                  ┌────────── Generic Space     │
  │                    ├──► Cross-Space ──┤                              │
  │  Input 2 ──────────┘     Mapping     │  ┌──────────────────────┐    │
  │                                      │  │    Blended Space     │    │
  │                                      │  │  ┌─── emergent ──┐  │    │
  │                                      │  │  │  structure    │  │    │
  │                                      │  └──┴───────────────┴──┘    │
  └──────────────────────────────────────────────────────────────────────┘

Key principles:
  [CB-1] Cross-Space Mapping: Partial structural mapping between input spaces
  [CB-2] Generic Space: Abstract structure shared by both inputs
  [CB-3] Blended Space: Selective projection + emergent structure
  [CB-4] Emergent Structure: Composition, completion, elaboration
  [CB-5] Vital Relations: Change, cause-effect, time, identity, etc.

Additional techniques:
  [CE-1] Constraint relaxation (assumption violation)
  [CE-2] Bisociation (Koestler's matrix crossing)
  [CE-3] Novelty evaluation (Schmidhuber's compression progress)
  [CE-4] Creative dreaming (offline memory recombination)
  [CE-5] Curiosity-driven exploration (epistemic foraging)
"""

import json
import math
import random
import threading
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set

import numpy as np

warnings.filterwarnings("ignore")

BRAIN_DIR = Path(__file__).parent.resolve()
DATA_FILE = BRAIN_DIR / "creativity_data.json"

# ── Configuration ───────────────────────────────────────────────────────────

MAX_JOURNAL_ENTRIES = 500
NOVELTY_DISTANCE_THRESHOLD = 0.3
BLEND_COMPATIBILITY_THRESHOLD = 0.4
RELAXATION_MAX_DEPTH = 5
BISECTION_MAX_CONNECTIONS = 10
CREATIVE_DREAM_COUNT = 5
IDEA_DECAY_DAYS = 30.0
MIN_FEASIBILITY_FOR_RANKING = 0.1

# Conceptual blending parameters
MAX_EMERGENT_CONTRADICTIONS = 10
BLEND_OPTIMIZATION_ITERATIONS = 100
VITAL_RELATIONS = ["CHANGE", "CAUSE_EFFECT", "TIME", "IDENTITY",
                    "CONTAINMENT", "ROLE", "PART_WHOLE", "REPRESENTATION",
                    "CATEGORY", "INTENTIONALITY", "SIMILARITY", "ANALOGY"]


def _timestamp() -> str:
    return datetime.now().isoformat()


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _feature_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    all_keys = set(a.keys()) | set(b.keys())
    if not all_keys:
        return 1.0
    sum_sq = 0.0
    for k in all_keys:
        va = a.get(k, 0.0)
        vb = b.get(k, 0.0)
        sum_sq += (va - vb) ** 2
    return min(1.0, math.sqrt(sum_sq / len(all_keys)))


# ── Data Structures ─────────────────────────────────────────────────────────


class InputSpace:
    """An input mental space for conceptual blending."""

    def __init__(self, name: str, domain: str = "general",
                 elements: Dict[str, Any] = None,
                 structure: List[Tuple[str, str, str]] = None):
        self.name = name
        self.domain = domain
        self.elements = elements or {}  # element_name → attributes
        self.structure = structure or []  # (source, relation, target)
        self.attributes = {}  # Additional metadata

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "domain": self.domain,
            "elements": self.elements,
            "structure": self.structure,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InputSpace":
        return cls(d["name"], d.get("domain", "general"),
                   d.get("elements", {}), d.get("structure", []))


class GenericSpace:
    """Abstract structure shared by input spaces."""

    def __init__(self):
        self.shared_relations: List[str] = []
        self.abstract_roles: List[str] = []
        self.abstract_properties: Dict[str, Any] = {}
        self.mapping: Dict[str, Dict[str, str]] = {}  # space → {element: abstract_role}


class Blend:
    """A conceptual blend from Fauconnier & Turner's network model."""

    def __init__(self, name: str):
        self.name = name
        self.inputs: List[InputSpace] = []
        self.generic_space = GenericSpace()
        self.cross_mapping: Dict[str, str] = {}  # input1_element → input2_element
        self.projected_elements: Set[str] = set()
        self.emergent_structure: List[Dict] = []
        self.contradictions: List[Dict] = []
        self.blend_score: float = 0.0
        self.optimality_principles: Dict[str, float] = {
            "integration": 0.0,
            "web": 0.0,
            "unpacking": 0.0,
            "topology": 0.0,
            "good_reason": 0.0,
            "metonymic_tightening": 0.0,
        }
        self.created_at = _timestamp()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "inputs": [i.to_dict() for i in self.inputs],
            "cross_mapping": self.cross_mapping,
            "projected_elements": list(self.projected_elements),
            "emergent_structure": self.emergent_structure,
            "contradictions": self.contradictions,
            "blend_score": self.blend_score,
            "optimality_principles": self.optimality_principles,
            "created_at": self.created_at,
        }


# ── Conceptual Blending Engine ──────────────────────────────────────────────


class ConceptualBlender:
    """
    Fauconnier & Turner's Conceptual Integration Network.

    Full implementation of the blending framework:
    1. Cross-space mapping between input spaces
    2. Generic space construction
    3. Selective projection to blend
    4. Emergent structure (composition, completion, elaboration)
    5. Optimization via optimality principles
    """

    def __init__(self):
        self._blend_history: List[Blend] = []

    def blend(self, input1: InputSpace, input2: InputSpace) -> Blend:
        """
        Full conceptual blending pipeline.

        Args:
            input1: First input mental space
            input2: Second input mental space

        Returns:
            Blend with emergent structure
        """
        blend = Blend(f"blend({input1.name}, {input2.name})")
        blend.inputs = [input1, input2]

        # Step 1: Cross-space mapping
        cross_mapping = self._build_cross_mapping(input1, input2)
        blend.cross_mapping = cross_mapping

        # Step 2: Generic space
        generic = self._build_generic_space(input1, input2, cross_mapping)
        blend.generic_space = generic

        # Step 3: Selective projection
        projected = self._selective_projection(input1, input2, cross_mapping)
        blend.projected_elements = projected

        # Step 4: Emergent structure
        emergent = self._generate_emergent_structure(
            input1, input2, cross_mapping, projected
        )
        blend.emergent_structure = emergent.get("emergent", [])
        blend.contradictions = emergent.get("contradictions", [])

        # Step 5: Optimize via optimality principles
        blend.optimality_principles = self._evaluate_optimality(
            blend, input1, input2
        )
        blend.blend_score = self._compute_blend_score(blend)

        self._blend_history.append(blend)
        if len(self._blend_history) > 100:
            self._blend_history = self._blend_history[-100:]

        return blend

    def _build_cross_mapping(self, input1: InputSpace,
                              input2: InputSpace) -> Dict[str, str]:
        """Build structural mapping between inputs."""
        mapping = {}

        # Match by element name similarity
        for elem1 in input1.elements:
            best_match = None
            best_score = 0.0
            for elem2 in input2.elements:
                score = self._element_similarity(
                    input1.elements[elem1], input2.elements[elem2]
                )
                if score > best_score and score > BLEND_COMPATIBILITY_THRESHOLD:
                    best_score = score
                    best_match = elem2

            if best_match and best_match not in mapping.values():
                mapping[elem1] = best_match

        # Match by structural role
        for rel1 in input1.structure:
            for rel2 in input2.structure:
                if rel1[1] == rel2[1]:  # Same relation name
                    if rel1[0] not in mapping and rel2[0] not in mapping.values():
                        mapping[rel1[0]] = rel2[0]
                    if rel1[2] not in mapping and rel2[2] not in mapping.values():
                        mapping[rel1[2]] = rel2[2]

        return mapping

    def _element_similarity(self, attrs1: Any, attrs2: Any) -> float:
        """Compute similarity between element attributes."""
        if isinstance(attrs1, dict) and isinstance(attrs2, dict):
            shared = set(attrs1.keys()) & set(attrs2.keys())
            if not shared:
                return 0.0
            matches = sum(1 for k in shared if attrs1[k] == attrs2[k])
            return matches / len(shared)
        return 0.3  # Default moderate similarity

    def _build_generic_space(self, input1: InputSpace, input2: InputSpace,
                              mapping: Dict[str, str]) -> GenericSpace:
        """Build abstract generic space from both inputs."""
        generic = GenericSpace()

        # Find shared relation types
        rel_names1 = {r[1] for r in input1.structure}
        rel_names2 = {r[1] for r in input2.structure}
        generic.shared_relations = list(rel_names1 & rel_names2)

        # Abstract roles from mapping
        for elem1, elem2 in mapping.items():
            role = f"abstract({elem1}_{elem2})"
            generic.abstract_roles.append(role)
            generic.mapping.setdefault("input1", {})[elem1] = role
            generic.mapping.setdefault("input2", {})[elem2] = role

        return generic

    def _selective_projection(self, input1: InputSpace, input2: InputSpace,
                               mapping: Dict[str, str]) -> Set[str]:
        """Selectively project elements to blend."""
        projected = set()

        # Project mapped elements
        for elem1, elem2 in mapping.items():
            projected.add(elem1)
            projected.add(elem2)

        # Project elements with high diagnostic value
        for elem in input1.elements:
            if elem not in projected:
                # Check if element is highly distinctive
                attrs = input1.elements.get(elem, {})
                if isinstance(attrs, dict) and attrs.get("diagnostic", 0) > 0.7:
                    projected.add(elem)

        for elem in input2.elements:
            if elem not in projected:
                attrs = input2.elements.get(elem, {})
                if isinstance(attrs, dict) and attrs.get("diagnostic", 0) > 0.7:
                    projected.add(elem)

        return projected

    def _generate_emergent_structure(self, input1: InputSpace, input2: InputSpace,
                                      mapping: Dict[str, str],
                                      projected: Set[str]) -> Dict:
        """Generate emergent structure from blending."""
        emergent = []
        contradictions = []

        # Composition: new relations from projections
        for proj1 in projected:
            if proj1 in input1.elements:
                for proj2 in projected:
                    if proj2 in input2.elements and proj2 != proj1:
                        # Check compatibility
                        if mapping.get(proj1) == proj2:
                            continue  # Already mapped
                        emergent.append({
                            "type": "composition",
                            "elements": [proj1, proj2],
                            "description": f"New pairing: {proj1} from {input1.name} "
                                           f"with {proj2} from {input2.name}",
                            "novelty": 0.8,
                        })

        # Completion: infer missing structure from background knowledge
        for rel1 in input1.structure:
            for rel2 in input2.structure:
                if rel1[1] == rel2[1]:  # Same relation
                    # Check if both participants are projected
                    if rel1[0] in projected and rel2[0] in projected:
                        emergent.append({
                            "type": "completion",
                            "relation": rel1[1],
                            "description": f"Shared relation '{rel1[1]}' between "
                                           f"{rel1[0]} and {rel2[0]}",
                            "novelty": 0.6,
                        })

        # Elaboration: run the blend dynamically
        elaboration = self._elaborate_blend(input1, input2, mapping, projected)
        emergent.extend(elaboration)

        # Detect contradictions
        for proj1 in projected:
            if proj1 in input1.elements and mapping.get(proj1) in input2.elements:
                mapped = mapping[proj1]
                attrs1 = input1.elements.get(proj1, {})
                attrs2 = input2.elements.get(mapped, {})
                if isinstance(attrs1, dict) and isinstance(attrs2, dict):
                    for k in set(attrs1) & set(attrs2):
                        if attrs1[k] != attrs2[k]:
                            contradictions.append({
                                "feature": k,
                                "value1": attrs1[k],
                                "value2": attrs2[k],
                                "resolution": "creative_tension",
                            })

        return {"emergent": emergent, "contradictions": contradictions}

    def _elaborate_blend(self, input1: InputSpace, input2: InputSpace,
                          mapping: Dict[str, str],
                          projected: Set[str]) -> List[Dict]:
        """Elaborate the blend by running dynamic simulations."""
        elaboration = []

        # Combine structural relations
        combined_structure = []
        for rel in input1.structure:
            if rel[0] in projected or rel[2] in projected:
                combined_structure.append(rel)
        for rel in input2.structure:
            if rel[0] in projected or rel[2] in projected:
                combined_structure.append(rel)

        # Generate compound relations
        for i, rel1 in enumerate(combined_structure):
            for rel2 in combined_structure[i+1:]:
                if rel1[2] == rel2[0]:  # Chain: X → Y → Z
                    elaboration.append({
                        "type": "elaboration",
                        "chain": f"{rel1[0]} --[{rel1[1]}]--> {rel1[2]} "
                                 f"--[{rel2[1]}]--> {rel2[2]}",
                        "description": f"Causal chain: {rel1[0]} affects {rel1[2]} "
                                       f"which affects {rel2[2]}",
                        "novelty": 0.7,
                    })

        return elaboration

    def _evaluate_optimality(self, blend: Blend, input1: InputSpace,
                              input2: InputSpace) -> Dict[str, float]:
        """Evaluate Fauconnier & Turner's optimality principles."""
        return {
            "integration": self._integration_score(blend),
            "web": self._web_score(blend, input1, input2),
            "unpacking": self._unpacking_score(blend),
            "topology": self._topology_score(blend, input1, input2),
            "good_reason": self._good_reason_score(blend, input1, input2),
            "metonymic_tightening": self._metonymic_score(blend),
        }

    def _integration_score(self, blend: Blend) -> float:
        """How well the blend forms a coherent scene."""
        if not blend.emergent_structure:
            return 0.0
        return min(1.0, len(blend.emergent_structure) * 0.1)

    def _web_score(self, blend: Blend, input1: InputSpace,
                   input2: InputSpace) -> float:
        """How well the blend maintains connections to inputs."""
        if not blend.cross_mapping:
            return 0.0
        total_inputs = len(input1.elements) + len(input2.elements)
        if total_inputs == 0:
            return 0.0
        return len(blend.cross_mapping) / total_inputs

    def _unpacking_score(self, blend: Blend) -> float:
        """How well the blend can be unpacked back to inputs."""
        return min(1.0, len(blend.cross_mapping) * 0.2)

    def _topology_score(self, blend: Blend, input1: InputSpace,
                        input2: InputSpace) -> float:
        """How well blend preserves input structure."""
        if not input1.structure and not input2.structure:
            return 0.5
        preserved = 0
        total = 0
        for rel in input1.structure:
            total += 1
            if rel[0] in blend.projected_elements or rel[2] in blend.projected_elements:
                preserved += 1
        for rel in input2.structure:
            total += 1
            if rel[0] in blend.projected_elements or rel[2] in blend.projected_elements:
                preserved += 1
        return preserved / max(total, 1)

    def _good_reason_score(self, blend: Blend, input1: InputSpace,
                           input2: InputSpace) -> float:
        """How well the blend is motivated by purpose."""
        return min(1.0, len(blend.emergent_structure) /
                   max(len(blend.contradictions) + 1, 1))

    def _metonymic_score(self, blend: Blend) -> float:
        """How well the blend uses metonymic tightening."""
        if not blend.emergent_structure:
            return 0.0
        chain_count = sum(1 for e in blend.emergent_structure
                          if e.get("type") == "completion")
        return min(1.0, chain_count * 0.15)

    def _compute_blend_score(self, blend: Blend) -> float:
        """Weighted average of optimality principle scores."""
        if not blend.optimality_principles:
            return 0.0
        weights = {
            "integration": 0.25,
            "web": 0.15,
            "unpacking": 0.10,
            "topology": 0.20,
            "good_reason": 0.20,
            "metonymic_tightening": 0.10,
        }
        total = 0.0
        weight_sum = 0.0
        for key, score in blend.optimality_principles.items():
            w = weights.get(key, 0.1)
            total += score * w
            weight_sum += w
        return total / weight_sum if weight_sum > 0 else 0.0


# ── Constraint Relaxation ───────────────────────────────────────────────────


class ConstraintRelaxation:
    """Systematically explores what happens when assumptions are relaxed."""

    def __init__(self):
        self._relaxation_history: List[Dict] = []

    def relax_constraints(self, problem: Dict, assumptions: List[Dict],
                           max_depth: int = RELAXATION_MAX_DEPTH) -> List[Dict]:
        """Generate relaxed variants by toggling/modifying assumptions."""
        variants = []
        depth = min(len(assumptions), max_depth)

        # Single-assumption relaxations
        for assumption in assumptions[:depth]:
            for mode in assumption.get("relax_modes", ["remove"]):
                variant = self._apply_relaxation(problem, assumption, mode)
                variant["relaxation_depth"] = 1
                variant["score"] = self._score_variant(variant, problem)
                variants.append(variant)

        # Pairwise relaxations
        for i in range(min(depth, 4)):
            for j in range(i + 1, min(depth, 4)):
                a1, a2 = assumptions[i], assumptions[j]
                mode1 = random.choice(a1.get("relax_modes", ["remove"]))
                mode2 = random.choice(a2.get("relax_modes", ["remove"]))
                variant = self._apply_relaxation(problem, a1, mode1)
                variant = self._apply_relaxation(variant, a2, mode2)
                variant["relaxation_depth"] = 2
                variant["score"] = self._score_variant(variant, problem)
                variants.append(variant)

        variants.sort(key=lambda v: v["score"], reverse=True)
        self._relaxation_history.extend(variants[:5])
        if len(self._relaxation_history) > 100:
            self._relaxation_history = self._relaxation_history[-100:]
        return variants

    def _apply_relaxation(self, problem: Dict, assumption: Dict,
                          mode: str) -> Dict:
        variant = {
            "original_problem": problem.get("name", problem.get("description", "unknown")),
            "relaxed_assumption": assumption["name"],
            "relaxation_mode": mode,
            "description": "",
            "modified_constraints": dict(problem.get("constraints", {})),
            "modified_parameters": dict(problem.get("parameters", {})),
        }
        a_name = assumption["name"]
        if mode == "remove":
            variant["description"] = f"What if we don't require '{a_name}'?"
            variant["modified_constraints"].pop(a_name, None)
        elif mode == "invert":
            variant["description"] = f"What if '{a_name}' were reversed?"
            if a_name in variant["modified_constraints"]:
                val = variant["modified_constraints"][a_name]
                if isinstance(val, (int, float)):
                    variant["modified_constraints"][a_name] = -val
                elif isinstance(val, bool):
                    variant["modified_constraints"][a_name] = not val
                else:
                    variant["modified_constraints"][a_name] = f"NOT({val})"
        elif mode == "weaken":
            variant["description"] = f"What if '{a_name}' were less strict?"
            if a_name in variant["modified_constraints"]:
                val = variant["modified_constraints"][a_name]
                if isinstance(val, (int, float)):
                    variant["modified_constraints"][a_name] = val * 0.5
        elif mode == "generalize":
            variant["description"] = f"What if '{a_name}' applied more broadly?"
        return variant

    def _score_variant(self, variant: Dict, original: Dict) -> float:
        orig_constraints = original.get("constraints", {})
        var_constraints = variant.get("modified_constraints", {})
        if orig_constraints:
            satisfied = sum(1 for k in orig_constraints if k in var_constraints)
            satisfaction = satisfied / len(orig_constraints)
        else:
            satisfaction = 0.5
        novelty = 0.3 + variant.get("relaxation_depth", 1) * 0.2
        mode = variant.get("relaxation_mode", "")
        if mode in ("invert", "generalize"):
            novelty += 0.15
        return round(0.4 * satisfaction + 0.6 * novelty, 4)


# ── Bisociation ────────────────────────────────────────────────────────────


class Bisociation:
    """Koestler's bisociation: connecting two unrelated frames of reference."""

    def __init__(self):
        self._bisociation_history: List[Dict] = []

    def bisociate(self, frame_a: Dict, frame_b: Dict) -> List[Dict]:
        """Find unexpected structural connections between two frames."""
        connections = []

        struct_a = frame_a.get("structure", {})
        struct_b = frame_b.get("structure", {})
        patterns_a = frame_a.get("patterns", [])
        patterns_b = frame_b.get("patterns", [])

        # Structural parallels
        for rel_a, desc_a in struct_a.items():
            for rel_b, desc_b in struct_b.items():
                similarity = self._structural_similarity(desc_a, desc_b)
                if 0.3 < similarity < 0.9:
                    connections.append({
                        "type": "structural_parallel",
                        "frame_a_element": rel_a,
                        "frame_b_element": rel_b,
                        "similarity": round(similarity, 4),
                        "insight": (
                            f"'{rel_a}' in {frame_a.get('name', 'A')} "
                            f"mirrors '{rel_b}' in {frame_b.get('name', 'B')}"
                        ),
                    })

        # Pattern inversions
        for pat_a in patterns_a:
            for pat_b in patterns_b:
                if self._are_inverted_patterns(pat_a, pat_b):
                    connections.append({
                        "type": "pattern_inversion",
                        "pattern_a": pat_a,
                        "pattern_b": pat_b,
                        "insight": f"Pattern '{pat_a}' and '{pat_b}' are inversions",
                        "surprise_score": 0.8,
                    })

        # Cross-domain mappings
        shared_abstractions = self._find_shared_abstractions(struct_a, struct_b)
        for abstraction in shared_abstractions:
            connections.append({
                "type": "cross_domain_mapping",
                "abstraction": abstraction,
                "insight": f"Both frames share abstract structure '{abstraction}'",
                "surprise_score": 0.7,
            })

        for conn in connections:
            conn["surprise_score"] = conn.get("surprise_score", 0.5)
            conn["overall_score"] = round(
                conn["surprise_score"] * 0.6 + conn.get("similarity", 0.5) * 0.4, 4
            )

        connections.sort(key=lambda c: c.get("overall_score", 0), reverse=True)
        top = connections[:BISECTION_MAX_CONNECTIONS]

        self._bisociation_history.extend(top)
        if len(self._bisociation_history) > 100:
            self._bisociation_history = self._bisociation_history[-100:]
        return top

    def _structural_similarity(self, desc_a: Any, desc_b: Any) -> float:
        if isinstance(desc_a, str) and isinstance(desc_b, str):
            words_a = set(desc_a.lower().split())
            words_b = set(desc_b.lower().split())
            if not words_a or not words_b:
                return 0.0
            intersection = words_a & words_b
            union = words_a | words_b
            return len(intersection) / len(union) if union else 0.0
        elif isinstance(desc_a, (int, float)) and isinstance(desc_b, (int, float)):
            return 1.0 / (1.0 + abs(desc_a - desc_b))
        return 0.0

    def _are_inverted_patterns(self, pat_a: str, pat_b: str) -> bool:
        inversions = {
            "growth": "decay", "expand": "contract", "build": "destroy",
            "connect": "separate", "attract": "repel", "heat": "cool",
            "accelerate": "decelerate", "amplify": "dampen", "open": "close",
        }
        a_lower = pat_a.lower()
        b_lower = pat_b.lower()
        for fwd, rev in inversions.items():
            if (fwd in a_lower and rev in b_lower) or (rev in a_lower and fwd in b_lower):
                return True
        return False

    def _find_shared_abstractions(self, struct_a: Dict, struct_b: Dict) -> List[str]:
        abstractions = []
        a_rels = set(struct_a.keys())
        b_rels = set(struct_b.keys())
        if len(a_rels) == len(b_rels) and len(a_rels) > 1:
            abstractions.append(f"isomorphic_{len(a_rels)}-element_structure")
        return abstractions


# ── Creativity Engine ──────────────────────────────────────────────────────


class CreativityEngine:
    """
    Central creativity engine combining conceptual blending, bisociation,
    constraint relaxation, and novelty evaluation.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.blender = ConceptualBlender()
        self.relaxer = ConstraintRelaxation()
        self.bisociator = Bisociation()
        self._journal: List[Dict] = []
        self._stats = {
            "ideas_generated": 0,
            "ideas_evaluated": 0,
            "dream_cycles_used": 0,
            "alternatives_generated": 0,
            "blends_performed": 0,
            "relaxations_performed": 0,
            "bisociations_performed": 0,
        }
        self._load()

    # ── Persistence ─────────────────────────────────────────────────────

    def _load(self):
        with self._lock:
            if DATA_FILE.exists():
                try:
                    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
                    self._journal = raw.get("journal", [])
                    self._stats.update(raw.get("stats", {}))
                    print(f"[CreativityEngine] Loaded {len(self._journal)} journal entries")
                except Exception as e:
                    print(f"[CreativityEngine] Load error: {e}")

    def _save(self):
        with self._lock:
            try:
                payload = {
                    "version": 2,
                    "updated": _timestamp(),
                    "journal": self._journal[-MAX_JOURNAL_ENTRIES:],
                    "stats": self._stats,
                }
                DATA_FILE.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            except Exception as e:
                print(f"[CreativityEngine] Save error: {e}")

    # ── Novelty Evaluation ─────────────────────────────────────────────

    def evaluate_novelty(self, idea: str) -> float:
        """Schmidhuber-inspired novelty assessment."""
        if not self._journal:
            return 0.7
        # Compare by word overlap (binary feature)
        idea_words = set(idea.lower().split())
        if not idea_words:
            return 0.7
        # Find max distance from any existing idea
        max_dist = 0.0
        for entry in self._journal:
            desc = entry.get("description", "")
            if not desc:
                continue
            desc_words = set(str(desc).lower().split())
            if not idea_words and not desc_words:
                continue
            overlap = len(idea_words & desc_words)
            union = len(idea_words | desc_words)
            jaccard = overlap / max(union, 1)
            max_dist = max(max_dist, 1.0 - jaccard)
        return min(1.0, max_dist if max_dist > 0 else 0.7)

    def _evaluate(self, idea: Dict, evaluator_data: Optional[List[Dict]] = None) -> Dict:
        """Evaluate idea on novelty, surprise, usefulness."""
        idea_features = idea.get("features", {})
        existing = evaluator_data or self._journal

        # Novelty
        if existing and idea_features:
            distances = [_feature_distance(idea_features, e.get("features", {}))
                        for e in existing]
            novelty = min(1.0, max(distances) if distances else 1.0)
        else:
            novelty = 0.7

        # Surprise
        if existing and idea_features:
            feature_means = defaultdict(float)
            feature_counts = defaultdict(int)
            for ex in existing:
                for k, v in ex.get("features", {}).items():
                    feature_means[k] += v
                    feature_counts[k] += 1
            for k in feature_means:
                feature_means[k] /= max(1, feature_counts[k])
            surprise_sum = 0.0
            surprise_count = 0
            for k, v in idea_features.items():
                if k in feature_means:
                    surprise_sum += abs(v - feature_means[k])
                    surprise_count += 1
            surprise = min(1.0, surprise_sum / max(1, surprise_count) * 2)
        else:
            surprise = 0.5

        usefulness = 0.5
        overall = round(0.4 * novelty + 0.3 * surprise + 0.3 * usefulness, 4)

        return {"novelty": novelty, "surprise": surprise, "usefulness": usefulness, "overall_score": overall}

    # ── Generate Alternatives ──────────────────────────────────────────

    def generate_alternatives(self, problem: Dict, existing_solution: Dict,
                               n: int = 5) -> List[Dict]:
        """Generate creative alternatives using all techniques."""
        with self._lock:
            alternatives = []

            # Technique 1: Conceptual blending
            input1 = InputSpace(existing_solution.get("name", "solution"),
                                elements=existing_solution.get("features", {}),
                                structure=existing_solution.get("structure", []))
            candidates = random.sample(self._journal, min(3, len(self._journal))) if self._journal else []
            for cand in candidates:
                cand_features = cand.get("features", {})
                input2 = InputSpace(cand.get("description", "candidate")[:30],
                                    elements=cand_features,
                                    structure=[])
                blend = self.blender.blend(input1, input2)
                alternatives.append({
                    "technique": "conceptual_blend",
                    "idea": blend.to_dict(),
                    "blend_score": blend.blend_score,
                    "feasibility": 0.6,
                })
                self._stats["blends_performed"] += 1

            # Technique 2: Constraint relaxation
            assumptions = problem.get("assumptions", [
                {"name": a, "relax_modes": ["remove", "invert"]}
                for a in problem.get("constraints", {}).keys()
            ])
            if assumptions:
                relaxed = self.relaxer.relax_constraints(problem, assumptions)
                for variant in relaxed[:3]:
                    alternatives.append({
                        "technique": "constraint_relaxation",
                        "idea": variant,
                        "feasibility": max(0.2, 1.0 - variant.get("relaxation_depth", 1) * 0.2),
                    })
                    self._stats["relaxations_performed"] += 1

            # Technique 3: Bisociation
            if len(self._journal) >= 2:
                other = random.choice(self._journal)
                connections = self.bisociator.bisociate(existing_solution, other)
                for conn in connections[:2]:
                    alternatives.append({
                        "technique": "bisociation",
                        "idea": conn,
                        "feasibility": 0.4,
                    })
                    self._stats["bisociations_performed"] += 1

            # Evaluate and rank
            for alt in alternatives:
                idea = alt["idea"]
                if isinstance(idea, dict):
                    idea_features = {}
                    if "elements" in idea:
                        idea_features = {k: 1.0 for k in idea.get("elements", {})}
                    elif "features" in idea:
                        idea_features = idea["features"]
                    elif "emergent_structure" in idea:
                        idea_features = {s.get("description", f"em_{i}"): 1.0
                                        for i, s in enumerate(idea.get("emergent_structure", []))}
                    eval_result = self._evaluate({"features": idea_features})
                    alt["novelty"] = eval_result["novelty"]
                    alt["surprise"] = eval_result["surprise"]
                    alt["rank_score"] = round(eval_result["novelty"] * alt["feasibility"], 4)

            alternatives.sort(key=lambda a: a["rank_score"], reverse=True)

            # Journal top results
            for alt in alternatives[:n]:
                self._journal_entry(alt["idea"], alt["technique"], alt["rank_score"])

            self._stats["alternatives_generated"] += len(alternatives[:n])
            self._stats["ideas_generated"] += len(alternatives[:n])
            self._save()

            return alternatives[:n]

    # ── Creative Critique ──────────────────────────────────────────────

    def creative_critique(self, idea: Dict) -> Dict:
        """Evaluate an idea's creativity."""
        with self._lock:
            idea_features = {}
            if isinstance(idea, dict):
                for k in ["desc", "features", "description", "insight"]:
                    if k in idea:
                        idea_features = {"content": 1.0 if isinstance(idea[k], str) and len(idea[k]) > 0 else 0.0}
                        break

            eval_result = self._evaluate({"features": idea_features})
            self._stats["ideas_evaluated"] += 1

            novelty = eval_result["novelty"]
            surprise = eval_result["surprise"]
            usefulness = eval_result["usefulness"]

            strengths = []
            weaknesses = []

            if novelty > 0.7:
                strengths.append("Highly novel")
            elif novelty > 0.4:
                strengths.append("Moderately novel")
            else:
                weaknesses.append("Low novelty")

            if surprise > 0.6:
                strengths.append("Surprising")
            else:
                weaknesses.append("Predictable")

            if usefulness > 0.6:
                strengths.append("Practical")
            elif usefulness > 0.3:
                strengths.append("Potentially useful")
            else:
                weaknesses.append("Unclear practical value")

            creativity_score = round(0.35 * novelty + 0.35 * surprise + 0.30 * usefulness, 4)

            if creativity_score > 0.7:
                verdict = "highly_creative"
            elif creativity_score > 0.5:
                verdict = "moderately_creative"
            elif creativity_score > 0.3:
                verdict = "mildly_creative"
            else:
                verdict = "conventional"

            return {
                "creativity_score": creativity_score,
                "verdict": verdict,
                "dimensions": eval_result,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "timestamp": _timestamp(),
            }

    # ── Creative Dreaming ──────────────────────────────────────────────

    def creative_dream(self, memories: Optional[List[Dict]] = None) -> List[Dict]:
        """Creative dreaming: combine memories from different categories."""
        with self._lock:
            if not memories:
                memories = self._journal
            if len(memories) < 2:
                return []

            by_category = defaultdict(list)
            for mem in memories:
                cat = mem.get("category", mem.get("technique", "unknown"))
                by_category[cat].append(mem)

            categories = list(by_category.keys())
            dream_ideas = []

            for _ in range(CREATIVE_DREAM_COUNT):
                if len(categories) < 2:
                    break
                cat_a, cat_b = random.sample(categories, 2)
                mem_a = random.choice(by_category[cat_a])
                mem_b = random.choice(by_category[cat_b])

                input1 = InputSpace(mem_a.get("description", "mem_a")[:30],
                                    elements=mem_a.get("features", {}))
                input2 = InputSpace(mem_b.get("description", "mem_b")[:30],
                                    elements=mem_b.get("features", {}))
                blend = self.blender.blend(input1, input2)

                dream_score = blend.blend_score
                dream_ideas.append({
                    "type": "creative_dream",
                    "source_categories": [cat_a, cat_b],
                    "idea": blend.to_dict(),
                    "depth": blend.blend_score,
                    "dream_score": dream_score,
                    "timestamp": _timestamp(),
                })

            for dream in dream_ideas:
                self._journal_entry(dream["idea"], "creative_dream", dream["dream_score"])

            self._stats["dream_cycles_used"] += 1
            self._stats["ideas_generated"] += len(dream_ideas)
            self._save()
            return dream_ideas

    def dream_integration(self, dream_memories: List[Dict]) -> List[Dict]:
        """Integration with dreaming system."""
        return self.creative_dream(dream_memories)

    def curiosity_integration(self, curiosity_targets: List[Dict]) -> List[Dict]:
        """Integration with curiosity module."""
        suggestions = []
        for target in curiosity_targets:
            topic = target.get("topic", target.get("name", "unknown"))
            interest = target.get("interest_level", 0.5)
            if interest > 0.6:
                problem = {
                    "name": topic,
                    "constraints": {},
                    "assumptions": [{"name": "conventional_approach", "relax_modes": ["invert", "remove"]}],
                }
                existing = {"name": f"standard_{topic}", "features": {}}
                alts = self.generate_alternatives(problem, existing, n=2)
                suggestions.extend(alts)
        return suggestions

    def world_model_integration(self, idea: Dict, world_model=None) -> Dict:
        """Evaluate creative ideas via simulation."""
        if world_model is not None and hasattr(world_model, "evaluate_plan"):
            try:
                action_sequence = [{"tool": idea.get("name", "idea"), "complexity": 0.5}]
                simulation = world_model.evaluate_plan(action_sequence)
                return {"idea": idea.get("name", "unnamed"), "simulated_outcome": simulation,
                        "evaluation_method": "world_model"}
            except Exception:
                pass
        critique = self.creative_critique(idea)
        return {"idea": idea.get("name", "unnamed"),
                "simulated_outcome": {"feasibility": critique["dimensions"]["usefulness"],
                                      "impact": critique["dimensions"]["novelty"]},
                "evaluation_method": "heuristic"}

    # ── Journal ────────────────────────────────────────────────────────

    def _journal_entry(self, idea: Dict, technique: str, score: float):
        entry = {
            "id": f"idea_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            "technique": technique,
            "features": idea.get("features", {}),
            "description": idea.get("name", idea.get("description", idea.get("insight", ""))),
            "score": score,
            "outcome": None,
            "created": _timestamp(),
        }
        self._journal.append(entry)
        if len(self._journal) > MAX_JOURNAL_ENTRIES:
            self._journal = self._journal[-MAX_JOURNAL_ENTRIES:]

    def record_outcome(self, idea_id: str, outcome: str, success: bool):
        with self._lock:
            for entry in self._journal:
                if entry.get("id") == idea_id:
                    entry["outcome"] = {"result": outcome, "success": success, "recorded": _timestamp()}
                    self._save()
                    return

    def get_journal(self, technique: Optional[str] = None, min_score: float = 0.0,
                    limit: int = 20) -> List[Dict]:
        with self._lock:
            entries = self._journal
            if technique:
                entries = [e for e in entries if e.get("technique") == technique]
            entries = [e for e in entries if e.get("score", 0) >= min_score]
            return entries[-limit:]

    def format_for_prompt(self, max_chars: int = 2000) -> str:
        with self._lock:
            lines = ["=== Creativity Engine (Conceptual Blending) ==="]
            lines.append(f"Journal entries: {len(self._journal)}")
            lines.append(f"Ideas generated: {self._stats['ideas_generated']}")
            lines.append(f"Dream cycles: {self._stats['dream_cycles_used']}")

            recent = sorted(self._journal, key=lambda x: x.get("score", 0), reverse=True)[:5]
            if recent:
                lines.append("\nTop ideas:")
                for entry in recent:
                    desc = entry.get("description", "")[:80]
                    score = entry.get("score", 0)
                    tech = entry.get("technique", "?")
                    lines.append(f"  [{tech}] {desc} (score: {score:.2f})")

            with_outcomes = [e for e in self._journal if e.get("outcome")]
            if with_outcomes:
                successes = sum(1 for e in with_outcomes if e["outcome"].get("success"))
                lines.append(f"\nOutcomes: {successes}/{len(with_outcomes)} successful")

            result = "\n".join(lines)
            if len(result) > max_chars:
                result = result[:max_chars - 3] + "..."
            return result

    def get_stats(self) -> Dict:
        with self._lock:
            journal_by_technique = defaultdict(int)
            for entry in self._journal:
                journal_by_technique[entry.get("technique", "unknown")] += 1
            outcomes = [e for e in self._journal if e.get("outcome")]
            success_rate = sum(1 for e in outcomes if e["outcome"].get("success")) / max(1, len(outcomes))
            return {**self._stats, "journal_size": len(self._journal),
                    "journal_by_technique": dict(journal_by_technique),
                    "success_rate": round(success_rate, 4)}


# ── Singleton ───────────────────────────────────────────────────────────────

_creativity_engine_instance: Optional[CreativityEngine] = None
_creativity_engine_lock = threading.Lock()


def get_creativity_engine() -> CreativityEngine:
    global _creativity_engine_instance
    if _creativity_engine_instance is None:
        with _creativity_engine_lock:
            if _creativity_engine_instance is None:
                _creativity_engine_instance = CreativityEngine()
    return _creativity_engine_instance


# ── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ce = get_creativity_engine()

    # Test conceptual blending
    input1 = InputSpace("solar_system", "physics",
                        elements={"sun": {"mass": "large", "temp": "hot"},
                                  "planet": {"orbits": True, "size": "small"}},
                        structure=[("sun", "ATTRACTS", "planet")])
    input2 = InputSpace("atom", "physics",
                        elements={"nucleus": {"mass": "large", "charge": "positive"},
                                  "electron": {"orbits": True, "size": "small"}},
                        structure=[("nucleus", "ATTRACTS", "electron")])

    blend = ce.blender.blend(input1, input2)
    print(f"Blend: {blend.name}")
    print(f"  Score: {blend.blend_score:.2f}")
    print(f"  Emergent structure: {len(blend.emergent_structure)}")
    print(f"  Contradictions: {len(blend.contradictions)}")
    print(f"  Optimality: {blend.optimality_principles}")

    # Test alternatives
    alternatives = ce.generate_alternatives(
        {"name": "test", "constraints": {"time": 10, "cost": 100}, "assumptions": []},
        {"name": "solution", "features": {"speed": 0.5, "quality": 0.7}},
        n=3
    )
    print(f"\nAlternatives: {len(alternatives)}")
    for a in alternatives:
        print(f"  [{a['technique']}] score={a['rank_score']:.2f}")

    print(f"\nStats: {ce.get_stats()}")
    print(f"Prompt:\n{ce.format_for_prompt()}")