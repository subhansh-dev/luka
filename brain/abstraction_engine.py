#!/usr/bin/env python3
"""
abstraction_engine.py — Luka Structure-Mapping Engine (SME)
=============================================================

Research-grade analogical reasoning engine based on Dedre Gentner's
Structure-Mapping Theory (SME/SME++).

Core Architecture (Gentner, 1983; Forbus & Gentner, 2017):
  ┌──────────────────────────────────────────────────────────────┐
  │                    STRUCTURE-MAPPING ENGINE                  │
  ├──────────────────────────────────────────────────────────────┤
  │  Representation: Propositional networks (entities + rels)   │
  │  Matching:      Systematicity-principled structural mapping │
  │  Inference:     Candidate Inferences from mapped structure  │
  │  Evaluation:    Structural evaluation scoring                │
  │  Transfer:      Cross-domain analogical problem solving     │
  └──────────────────────────────────────────────────────────────┘

Key principles:
  [SME-1] Systematicity: Deep systems of connected relations > isolated matches
  [SME-2] 1:1 Mapping: Each element maps to at most one element (no mergers)
  [SME-3] Parallel Connectivity: Relations map to same-arity relations
  [SME-4] Tiered Identicality: Higher-order relations > lower-order
  [SME-5] Candidate Inference: Unmatched structure in base predicts target

Additional capabilities:
  [AE-1] First-principles decomposition (assumption stripping)
  [AE-2] Counterfactual reasoning (what-if scenarios)
  [AE-3] Causal chain tracing (cause-effect propagation)
  [AE-4] Cross-domain transfer (solution migration)
  [AE-5] Emergent insight generation (conceptual blending)
  [AE-6] Abstract pattern recognition (schema induction)
"""

import hashlib
import json
import math
import random
import threading
import time
import warnings
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

warnings.filterwarnings("ignore")

BRAIN_DIR = Path(__file__).parent.resolve()
ABSTRACTION_FILE = BRAIN_DIR / "abstraction_state.json"

# ── Configuration ───────────────────────────────────────────────────────────

MAX_DOMAINS = 100
MAX_ANALOGIES = 500
MAX_PATTERNS = 300
MAX_INSIGHTS = 200
MAX_CROSS_TRANSFERS = 200
STRUCTURAL_SIMILARITY_THRESHOLD = 0.3
ANALOGY_MIN_RELATIONS = 2
DEFAULT_CHAIN_DEPTH = 5
MAX_CHAIN_DEPTH = 10
SME_SYSTEMATICITY_WEIGHT = 2.0  # Higher-order relations weighted more
SME_IDENTITY_PENALTY = 0.5  # Penalty for non-identical predicate matches
SME_MAX_MATCHES = 1000  # Max matching hypotheses to explore
MAX_HISTORY = 500


def _now() -> str:
    return datetime.now().isoformat()


def _timestamp() -> float:
    return time.time()


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


# ── SME Data Structures ─────────────────────────────────────────────────────


class Entity:
    """Represents an entity or object in a domain."""

    def __init__(self, name: str, attributes: Dict[str, Any] = None):
        self.name = name
        self.attributes = attributes or {}

    def __repr__(self):
        return f"Entity({self.name})"

    def to_dict(self) -> dict:
        return {"name": self.name, "attributes": self.attributes}

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(d["name"], d.get("attributes", {}))


class Relation:
    """A predicate/relation: e.g., CAUSE(A, B), ATTR(red, ball)."""

    def __init__(self, name: str, args: List[Union[str, "Relation"]],
                 order: int = 1):
        """
        Args:
            name: relation name (e.g., "CAUSE", "GREATER_THAN", "PULL")
            args: list of arguments (entities or nested relations)
            order: 1 = first-order, 2 = second-order, etc.
        """
        self.name = name
        self.args = args
        self.order = order

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"

    def to_dict(self) -> dict:
        args = []
        for a in self.args:
            if isinstance(a, Relation):
                args.append(a.to_dict())
            else:
                args.append({"entity": a})
        return {"name": self.name, "args": args, "order": self.order}

    @classmethod
    def from_dict(cls, d: dict) -> "Relation":
        args = []
        for a in d.get("args", []):
            if "entity" in a:
                args.append(a["entity"])
            elif "name" in a:
                args.append(Relation.from_dict(a))
        return cls(d["name"], args, d.get("order", 1))


class Domain:
    """A knowledge domain with entities and relations."""

    def __init__(self, name: str):
        self.name = name
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.properties: Dict[str, Any] = {}
        self.created_at = _now()
        self.last_used_at = _now()

    def add_entity(self, name: str, attributes: Dict[str, Any] = None) -> Entity:
        if name not in self.entities:
            self.entities[name] = Entity(name, attributes)
        return self.entities[name]

    def add_relation(self, name: str, args: List[Union[str, "Relation"]],
                     order: int = 1) -> Relation:
        rel = Relation(name, args, order)
        self.relations.append(rel)
        return rel

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "relations": [r.to_dict() for r in self.relations],
            "properties": self.properties,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Domain":
        dom = cls(d.get("name", "unknown"))
        for k, v in d.get("entities", {}).items():
            dom.entities[k] = Entity.from_dict(v)
        dom.relations = [Relation.from_dict(r) for r in d.get("relations", [])]
        dom.properties = d.get("properties", {})
        dom.created_at = d.get("created_at", _now())
        dom.last_used_at = d.get("last_used_at", _now())
        return dom


class MatchHypothesis:
    """A single candidate mapping between two relation structures."""

    def __init__(self, base_rel: Relation, target_rel: Relation,
                 arg_mappings: Dict[str, str], score: float = 0.0):
        self.base_rel = base_rel
        self.target_rel = target_rel
        self.arg_mappings = arg_mappings  # base_entity → target_entity
        self.score = score
        self.structural_eval = 0.0
        self.candidate_inferences: List[Relation] = []

    def __repr__(self):
        return f"Match({self.base_rel} ↔ {self.target_rel}, score={self.score:.2f})"


class Analogy:
    """A complete structural analogy between two domains."""

    def __init__(self, base_domain: str, target_domain: str):
        self.base_domain = base_domain
        self.target_domain = target_domain
        self.mappings: Dict[str, str] = {}  # base_entity → target_entity
        self.match_hypotheses: List[MatchHypothesis] = []
        self.candidate_inferences: List[Relation] = []
        self.structural_similarity: float = 0.0
        self.systematicity: float = 0.0
        self.analogy_id = _hash(f"{base_domain}:{target_domain}:{_now()}")
        self.created_at = _now()
        self.use_count = 0
        self.success_count = 0

    def to_dict(self) -> dict:
        return {
            "analogy_id": self.analogy_id,
            "base_domain": self.base_domain,
            "target_domain": self.target_domain,
            "mappings": self.mappings,
            "structural_similarity": round(self.structural_similarity, 4),
            "systematicity": round(self.systematicity, 4),
            "candidate_inferences": [r.to_dict() for r in self.candidate_inferences],
            "created_at": self.created_at,
            "use_count": self.use_count,
            "success_count": self.success_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Analogy":
        a = cls(d.get("base_domain", ""), d.get("target_domain", ""))
        a.analogy_id = d.get("analogy_id", a.analogy_id)
        a.mappings = d.get("mappings", {})
        a.structural_similarity = d.get("structural_similarity", 0.0)
        a.systematicity = d.get("systematicity", 0.0)
        a.candidate_inferences = [Relation.from_dict(r) for r in d.get("candidate_inferences", [])]
        a.created_at = d.get("created_at", _now())
        a.use_count = d.get("use_count", 0)
        a.success_count = d.get("success_count", 0)
        return a


# ── Structure-Mapping Engine (SME) ──────────────────────────────────────────


class StructureMappingEngine:
    """
    Gentner's Structure-Mapping Engine (SME).

    Finds structural analogies between two domains by:
    1. Computing local match hypotheses (identical predicates)
    2. Combining consistent hypotheses into global mappings
    3. Evaluating mappings by systematicity (connectedness)
    4. Generating candidate inferences (projecting unmapped structure)
    """

    def __init__(self):
        self._match_history: List[Dict] = []

    def find_analogies(self, base_domain: Domain,
                       target_domain: Domain) -> List[Analogy]:
        """
        Find structural analogies between base and target.

        Args:
            base_domain: Source domain with established structure
            target_domain: Target domain to map onto

        Returns:
            List of Analogy objects, ranked by structural similarity
        """
        # Step 1: Generate local match hypotheses
        match_hypotheses = self._generate_match_hypotheses(base_domain, target_domain)

        if not match_hypotheses:
            return []

        # Step 2: Combine into consistent global mappings
        global_mappings = self._combine_mappings(match_hypotheses)

        if not global_mappings:
            return []

        # Step 3: Evaluate mappings by systematicity
        evaluated = self._evaluate_mappings(global_mappings, base_domain, target_domain)

        # Step 4: Generate candidate inferences and build analogies
        analogies = []
        for mapping in evaluated[:3]:  # Top 3
            analogy = self._build_analogy(mapping, base_domain, target_domain)
            if analogy.structural_similarity >= STRUCTURAL_SIMILARITY_THRESHOLD:
                analogies.append(analogy)

        analogies.sort(key=lambda a: a.structural_similarity * a.systematicity, reverse=True)

        self._match_history.append({
            "base": base_domain.name,
            "target": target_domain.name,
            "matches": len(match_hypotheses),
            "analogies": len(analogies),
            "timestamp": _now(),
        })

        return analogies

    def _generate_match_hypotheses(self, base: Domain,
                                    target: Domain) -> List[MatchHypothesis]:
        """
        Generate local match hypotheses between identical predicates.

        SME Rule: Relations must have the same name to match.
        Higher-order relations (same order) are preferred.
        """
        hypotheses = []

        for b_rel in base.relations:
            for t_rel in target.relations:
                # Must have same name
                if b_rel.name != t_rel.name:
                    continue

                # Must have same arity
                if len(b_rel.args) != len(t_rel.args):
                    continue

                # Must have same order
                if b_rel.order != t_rel.order:
                    continue

                # Generate argument mappings
                arg_mappings = self._map_arguments(b_rel, t_rel, base, target)

                if arg_mappings is not None:
                    # Score the match
                    score = self._score_match(b_rel, t_rel, arg_mappings, base, target)

                    hypothesis = MatchHypothesis(b_rel, t_rel, arg_mappings, score)
                    hypotheses.append(hypothesis)

        # Sort by score descending
        hypotheses.sort(key=lambda h: h.score, reverse=True)

        # Limit to max matches
        return hypotheses[:SME_MAX_MATCHES]

    def _map_arguments(self, b_rel: Relation, t_rel: Relation,
                        base: Domain, target: Domain) -> Optional[Dict[str, str]]:
        """
        Map arguments between base and target relations.

        Returns dict {base_entity: target_entity} or None if incompatible.
        """
        mapping = {}

        for b_arg, t_arg in zip(b_rel.args, t_rel.args):
            if isinstance(b_arg, Relation) and isinstance(t_arg, Relation):
                # Nested relations — map recursively
                inner_mapping = self._map_arguments(b_arg, t_arg, base, target)
                if inner_mapping is None:
                    return None
                mapping.update(inner_mapping)

            elif isinstance(b_arg, str) and isinstance(t_arg, str):
                # Both are entities — check if they already have a mapping
                if b_arg in mapping:
                    if mapping[b_arg] != t_arg:
                        return None  # 1:1 mapping violation
                else:
                    # Check if target entity is already mapped to a different base entity
                    for k, v in mapping.items():
                        if v == t_arg and k != b_arg:
                            return None  # 1:1 mapping violation
                    mapping[b_arg] = t_arg
            else:
                # Type mismatch
                return None

        return mapping

    def _score_match(self, b_rel: Relation, t_rel: Relation,
                      arg_mappings: Dict[str, str],
                      base: Domain, target: Domain) -> float:
        """Score a match hypothesis."""
        score = 1.0

        # Systematicity bonus: higher-order relations weigh more
        score *= (1.0 + (b_rel.order - 1) * SME_SYSTEMATICITY_WEIGHT)

        # Attribute similarity bonus
        for b_entity, t_entity in arg_mappings.items():
            b_attrs = base.entities.get(b_entity, Entity(b_entity)).attributes
            t_attrs = target.entities.get(t_entity, Entity(t_entity)).attributes
            if b_attrs and t_attrs:
                shared = set(b_attrs.keys()) & set(t_attrs.keys())
                if shared:
                    matches = sum(1 for k in shared if b_attrs[k] == t_attrs[k])
                    score += 0.5 * (matches / max(len(shared), 1))

        return score

    def _combine_mappings(self, hypotheses: List[MatchHypothesis]) -> List[Dict[str, str]]:
        """
        Combine consistent match hypotheses into global mappings.

        Uses greedy algorithm: start with best match, then add compatible ones.
        """
        # Sort by score
        hypotheses.sort(key=lambda h: h.score, reverse=True)

        # Build global mappings
        global_mappings = []

        for hypothesis in hypotheses:
            # Check if this hypothesis is compatible with any existing mapping
            added = False
            for mapping in global_mappings:
                if self._is_compatible(hypothesis.arg_mappings, mapping):
                    mapping.update(hypothesis.arg_mappings)
                    added = True
                    break

            if not added:
                # Start new mapping
                global_mappings.append(dict(hypothesis.arg_mappings))

        # Prune redundant mappings
        unique_mappings = []
        seen = set()
        for m in global_mappings:
            frozen = tuple(sorted(m.items()))
            if frozen not in seen:
                seen.add(frozen)
                unique_mappings.append(m)

        return unique_mappings

    def _is_compatible(self, new_mapping: Dict[str, str],
                       existing_mapping: Dict[str, str]) -> bool:
        """Check if new mapping is compatible with existing mapping."""
        for b_entity, t_entity in new_mapping.items():
            if b_entity in existing_mapping:
                if existing_mapping[b_entity] != t_entity:
                    return False  # 1:1 violation
            for k, v in existing_mapping.items():
                if v == t_entity and k != b_entity:
                    return False  # 1:1 violation
        return True

    def _evaluate_mappings(self, mappings: List[Dict[str, str]],
                            base: Domain, target: Domain) -> List[Dict[str, Any]]:
        """Evaluate mappings by systematicity and structural coverage."""
        evaluated = []

        for mapping in mappings:
            # Count relations that are fully mapped
            mapped_relations = 0
            total_relations = 0
            systematicity = 0.0

            for b_rel in base.relations:
                total_relations += 1
                if self._is_relation_mapped(b_rel, mapping):
                    mapped_relations += 1
                    # Higher-order relations contribute more to systematicity
                    systematicity += b_rel.order

            # Also check target relations
            mapped_target = 0
            for t_rel in target.relations:
                if self._is_relation_mapped_target(t_rel, mapping):
                    mapped_target += 1
                    systematicity += t_rel.order

            structural_coverage = mapped_relations / max(total_relations, 1)
            target_coverage = mapped_target / max(len(target.relations), 1)

            # Normalize systematicity
            max_sys = sum(r.order for r in base.relations) + sum(r.order for r in target.relations)
            systematicity = systematicity / max(max_sys, 1) * 2.0

            # Structural similarity = coverage + systematicity
            structural_sim = (structural_coverage + target_coverage) / 2.0

            evaluated.append({
                "mapping": mapping,
                "structural_similarity": structural_sim,
                "systematicity": systematicity,
                "structural_coverage": structural_coverage,
                "target_coverage": target_coverage,
                "mapped_relations": mapped_relations,
                "total_relations": total_relations,
            })

        # Sort by structural similarity × systematicity
        evaluated.sort(key=lambda e: e["structural_similarity"] * e["systematicity"], reverse=True)
        return evaluated

    def _is_relation_mapped(self, relation: Relation,
                             mapping: Dict[str, str]) -> bool:
        """Check if all arguments of a relation are mapped."""
        for arg in relation.args:
            if isinstance(arg, Relation):
                if not self._is_relation_mapped(arg, mapping):
                    return False
            elif isinstance(arg, str):
                if arg not in mapping:
                    return False
        return True

    def _is_relation_mapped_target(self, relation: Relation,
                                    mapping: Dict[str, str]) -> bool:
        """Check if all arguments of a target relation are mapped."""
        reversed_mapping = {v: k for k, v in mapping.items()}
        for arg in relation.args:
            if isinstance(arg, Relation):
                if not self._is_relation_mapped_target(arg, mapping):
                    return False
            elif isinstance(arg, str):
                if arg not in reversed_mapping:
                    return False
        return True

    def _build_analogy(self, evaluation: Dict, base: Domain,
                        target: Domain) -> Analogy:
        """Build Analogy object from evaluation."""
        analogy = Analogy(base.name, target.name)
        analogy.mappings = evaluation["mapping"]
        analogy.structural_similarity = evaluation["structural_similarity"]
        analogy.systematicity = evaluation["systematicity"]

        # Generate candidate inferences
        analogy.candidate_inferences = self._generate_candidate_inferences(
            base, target, evaluation["mapping"]
        )

        return analogy

    def _generate_candidate_inferences(self, base: Domain, target: Domain,
                                        mapping: Dict[str, str]) -> List[Relation]:
        """
        Generate candidate inferences: unmapped base structure projected to target.

        These are the "new knowledge" from the analogy.
        """
        inferences = []

        # Find unmapped base relations
        for b_rel in base.relations:
            if self._contains_unmapped_entity(b_rel, mapping):
                continue

            # Check if this relation is already mapped to a target relation
            already_mapped = False
            for t_rel in target.relations:
                if b_rel.name == t_rel.name:
                    same_args = all(
                        mapping.get(a1, "") == a2 if isinstance(a1, str) and isinstance(a2, str)
                        else False
                        for a1, a2 in zip(b_rel.args, t_rel.args)
                    )
                    if same_args:
                        already_mapped = True
                        break

            if not already_mapped:
                # Project to target domain
                inference = self._project_to_target(b_rel, mapping)
                if inference:
                    inferences.append(inference)

        return inferences

    def _contains_unmapped_entity(self, relation: Relation,
                                   mapping: Dict[str, str]) -> bool:
        """Check if any argument is unmapped."""
        for arg in relation.args:
            if isinstance(arg, Relation):
                if self._contains_unmapped_entity(arg, mapping):
                    return True
            elif isinstance(arg, str):
                if arg not in mapping:
                    return True
        return False

    def _project_to_target(self, relation: Relation,
                            mapping: Dict[str, str]) -> Optional[Relation]:
        """Project a base relation to the target domain via mapping."""
        target_args = []
        for arg in relation.args:
            if isinstance(arg, Relation):
                projected = self._project_to_target(arg, mapping)
                if projected:
                    target_args.append(projected)
                else:
                    return None
            elif isinstance(arg, str):
                if arg in mapping:
                    target_args.append(mapping[arg])
                else:
                    return None

        return Relation(relation.name, target_args, relation.order)

    def get_stats(self) -> dict:
        return {
            "total_matches": sum(h.get("matches", 0) for h in self._match_history),
            "total_analogies": sum(h.get("analogies", 0) for h in self._match_history),
            "sessions": len(self._match_history),
        }


# ── Abstraction Engine ──────────────────────────────────────────────────────


class AbstractionEngine:
    """
    Cross-domain reasoning and abstraction with SME core.

    Integrates:
    - SME: analogical reasoning via structure mapping
    - First-principles decomposition
    - Counterfactual reasoning
    - Causal chain tracing
    - Cross-domain transfer
    - Emergent insight generation
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._domains: Dict[str, Domain] = {}
        self._analogies: List[Analogy] = []
        self._sme = StructureMappingEngine()
        self._insights: List[dict] = []
        self._transfers: List[dict] = []
        self._patterns: List[dict] = []
        self._counterfactuals: List[dict] = []
        self._first_principles: List[dict] = []
        self._meta: Dict[str, Any] = {
            "version": 2,
            "created": _now(),
            "last_update": _now(),
            "total_analogies": 0,
            "total_insights": 0,
            "total_transfers": 0,
            "total_counterfactuals": 0,
            "total_first_principles": 0,
        }
        self._load()

    # ── Persistence ─────────────────────────────────────────────────────

    def _load(self):
        if not ABSTRACTION_FILE.exists():
            self._save()
            return
        try:
            raw = ABSTRACTION_FILE.read_text(encoding="utf-8")
            data = json.loads(raw)
            self._meta = data.get("meta", self._meta)
            for name, d_dict in data.get("domains", {}).items():
                self._domains[name] = Domain.from_dict(d_dict)
            self._analogies = [Analogy.from_dict(a) for a in data.get("analogies", [])]
            self._insights = data.get("insights", [])
            self._transfers = data.get("transfers", [])
            self._counterfactuals = data.get("counterfactuals", [])
            self._first_principles = data.get("first_principles", [])
        except (json.JSONDecodeError, IOError):
            self._save()

    def _save(self):
        BRAIN_DIR.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "meta": self._meta,
                "domains": {name: d.to_dict() for name, d in self._domains.items()},
                "analogies": [a.to_dict() for a in self._analogies[-MAX_ANALOGIES:]],
                "insights": self._insights[-MAX_INSIGHTS:],
                "transfers": self._transfers[-MAX_CROSS_TRANSFERS:],
                "counterfactuals": self._counterfactuals[-MAX_HISTORY:],
                "first_principles": self._first_principles[-MAX_HISTORY:],
            }
            data["meta"]["last_update"] = _now()
            ABSTRACTION_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # ── Domain Management ───────────────────────────────────────────────

    def register_domain(self, name: str, entities: Dict[str, Dict] = None,
                        relations: List[Tuple[str, List[str], int]] = None,
                        properties: Dict[str, Any] = None) -> Domain:
        """Register a domain with entities and relations."""
        with self._lock:
            if name not in self._domains:
                self._domains[name] = Domain(name)
            dom = self._domains[name]

            if entities:
                for ename, attrs in entities.items():
                    dom.add_entity(ename, attrs)

            if relations:
                for rname, args, order in relations:
                    dom.add_relation(rname, args, order)

            if properties:
                dom.properties.update(properties)

            dom.last_used_at = _now()
            self._save()
            return dom

    def get_domain(self, name: str) -> Optional[Domain]:
        with self._lock:
            dom = self._domains.get(name)
            if dom:
                dom.last_used_at = _now()
            return dom

    def list_domains(self) -> List[str]:
        with self._lock:
            return list(self._domains.keys())

    # ── SME: Find Analogies ─────────────────────────────────────────────

    def find_analogies(self, base_domain: str,
                       target_domain: str) -> List[dict]:
        """Find structural analogies between two domains."""
        with self._lock:
            base = self._domains.get(base_domain)
            target = self._domains.get(target_domain)

            if not base or not target:
                return []

            analogies = self._sme.find_analogies(base, target)

            results = []
            for analogy in analogies:
                self._analogies.append(analogy)
                self._meta["total_analogies"] += 1

                # Generate explanation
                explanation = self._generate_analogy_explanation(analogy, base, target)

                results.append({
                    "analogy_id": analogy.analogy_id,
                    "base": base_domain,
                    "target": target_domain,
                    "mappings": analogy.mappings,
                    "structural_similarity": analogy.structural_similarity,
                    "systematicity": analogy.systematicity,
                    "candidate_inferences": [r.to_dict() for r in analogy.candidate_inferences],
                    "explanation": explanation,
                })

            self._save()
            return results

    def _generate_analogy_explanation(self, analogy: Analogy,
                                       base: Domain, target: Domain) -> str:
        """Generate human-readable explanation."""
        parts = [
            f"Structural analogy between '{base.name}' and '{target.name}':",
        ]

        if analogy.mappings:
            for b_entity, t_entity in list(analogy.mappings.items())[:5]:
                parts.append(f"  {b_entity} ↔ {t_entity}")

        if analogy.candidate_inferences:
            parts.append("Candidate inferences (new knowledge):")
            for inf in analogy.candidate_inferences[:3]:
                parts.append(f"  → {inf}")

        return "\n".join(parts)

    # ── First Principles ────────────────────────────────────────────────

    def first_principles(self, problem: str) -> dict:
        """Decompose problem to fundamental components."""
        with self._lock:
            assumptions = self._extract_assumptions(problem)
            constraints = self._identify_constraints(problem)
            components = self._decompose_to_components(problem)

            knowns = [c for c in components if c.get("confidence", 0) > 0.6]
            unknowns = [c for c in components if c.get("confidence", 0) <= 0.6]

            solution_space = self._rebuild_from_fundamentals(
                components, constraints, assumptions
            )

            result = {
                "problem": problem,
                "assumptions": assumptions,
                "constraints": constraints,
                "components": components,
                "knowns": knowns,
                "unknowns": unknowns,
                "solution_space": solution_space,
                "timestamp": _now(),
            }

            self._first_principles.append({
                "problem": problem[:200],
                "num_assumptions": len(assumptions),
                "num_components": len(components),
                "timestamp": _now(),
            })
            self._meta["total_first_principles"] += 1
            self._save()
            return result

    def _extract_assumptions(self, problem: str) -> List[dict]:
        """Extract hidden assumptions from problem description."""
        assumptions = []
        assumption_markers = [
            ("everyone knows", "common_belief"),
            ("obviously", "unexamined_premise"),
            ("always", "overgeneralization"),
            ("never", "overgeneralization"),
            ("impossible", "limiting_belief"),
            ("can't", "limiting_belief"),
            ("should", "normative_assumption"),
            ("must", "normative_assumption"),
            ("have to", "normative_assumption"),
            ("because", "causal_assumption"),
            ("is the only", "exclusivity_assumption"),
        ]

        problem_lower = problem.lower()
        for marker, atype in assumption_markers:
            if marker in problem_lower:
                idx = problem_lower.index(marker)
                context = problem[max(0, idx - 30):idx + len(marker) + 50]
                assumptions.append({
                    "type": atype,
                    "marker": marker,
                    "context": context.strip(),
                    "question": f"What if {marker} isn't true?",
                })

        return assumptions

    def _identify_constraints(self, problem: str) -> List[dict]:
        constraints = []
        problem_lower = problem.lower()
        constraint_patterns = [
            ("time", "temporal"), ("money", "financial"), ("resource", "resource"),
            ("memory", "computational"), ("bandwidth", "computational"),
            ("energy", "physical"), ("speed", "physical"), ("law", "legal"),
            ("regulation", "regulatory"), ("physics", "physical"),
            ("gravity", "physical"), ("thermodynamic", "physical"), ("entropy", "physical"),
        ]
        for keyword, ctype in constraint_patterns:
            if keyword in problem_lower:
                constraints.append({
                    "type": ctype, "keyword": keyword,
                    "description": f"Constraint related to: {keyword}",
                })
        return constraints

    def _decompose_to_components(self, problem: str) -> List[dict]:
        components = []
        sentences = [s.strip() for s in problem.replace("?", ".").split(".") if s.strip()]
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            component = {
                "id": f"c{i}",
                "statement": sentence,
                "word_count": len(words),
                "complexity": "high" if len(words) > 15 else "medium" if len(words) > 7 else "low",
                "confidence": 0.8 if len(words) > 3 else 0.4,
                "dependencies": [],
            }
            for j, other in enumerate(sentences):
                if i != j:
                    shared = set(sentence.lower().split()) & set(other.lower().split())
                    if len(shared) >= 2:
                        component["dependencies"].append(f"c{j}")
            components.append(component)
        return components

    def _rebuild_from_fundamentals(self, components: List[dict],
                                    constraints: List[dict],
                                    assumptions: List[dict]) -> List[dict]:
        solutions = []
        for assumption in assumptions[:3]:
            solutions.append({
                "strategy": "assumption_removal",
                "removed_assumption": assumption.get("marker", ""),
                "new_possibility": f"If we reject '{assumption.get('marker', '')}', "
                                   f"we could explore: {assumption.get('question', '')}",
                "novelty": 0.7,
            })
        for constraint in constraints[:2]:
            solutions.append({
                "strategy": "constraint_relaxation",
                "relaxed_constraint": constraint.get("keyword", ""),
                "new_possibility": f"What if {constraint.get('keyword', '')} were not a limitation?",
                "novelty": 0.6,
            })
        if len(components) >= 2:
            for i in range(min(3, len(components) - 1)):
                solutions.append({
                    "strategy": "recombination",
                    "components": [components[i]["id"], components[i + 1]["id"]],
                    "new_possibility": f"Combine: '{components[i]['statement'][:50]}' "
                                       f"with '{components[i+1]['statement'][:50]}'",
                    "novelty": 0.5,
                })
        return solutions

    # ── Counterfactual ──────────────────────────────────────────────────

    def counterfactual(self, scenario: str, change: str) -> dict:
        """Explore "what if X had been different?" reasoning."""
        with self._lock:
            direct_effects = self._trace_direct_effects(change)
            downstream = self._trace_downstream_effects(direct_effects)
            divergences = self._identify_divergences(scenario, change, direct_effects)
            plausibility = self._assess_counterfactual_plausibility(
                scenario, change, direct_effects, downstream
            )
            alternatives = self._generate_alternative_outcomes(scenario, change, downstream)

            result = {
                "original_scenario": scenario,
                "hypothetical_change": change,
                "direct_effects": direct_effects,
                "downstream_effects": downstream,
                "divergence_points": divergences,
                "plausibility": plausibility,
                "alternative_outcomes": alternatives,
                "timestamp": _now(),
            }

            self._counterfactuals.append({
                "scenario": scenario[:200],
                "change": change[:200],
                "num_effects": len(direct_effects) + len(downstream),
                "plausibility": plausibility,
                "timestamp": _now(),
            })
            self._meta["total_counterfactuals"] += 1
            self._save()
            return result

    def _trace_direct_effects(self, change: str) -> List[dict]:
        effects = []
        change_lower = change.lower()
        for dom in self._domains.values():
            for rel in dom.relations:
                desc = str(rel).lower()
                change_words = set(change_lower.split())
                desc_words = set(desc.split())
                overlap = change_words & desc_words
                if len(overlap) >= max(1, min(len(change_words), len(desc_words)) // 3):
                    effects.append({
                        "effect": str(rel),
                        "domain": dom.name,
                        "strength": min(1.0, len(overlap) / max(len(change_words), 1)),
                        "confidence": 0.5,
                    })
        return effects

    def _trace_downstream_effects(self, direct_effects: List[dict],
                                   depth: int = 3) -> List[dict]:
        downstream = []
        visited = set()
        frontier = [e["effect"] for e in direct_effects]
        for d in range(depth):
            next_frontier = []
            for effect_name in frontier:
                if effect_name in visited:
                    continue
                visited.add(effect_name)
                for dom in self._domains.values():
                    for rel in dom.relations:
                        desc = str(rel).lower()
                        if effect_name.lower() in desc:
                            downstream.append({
                                "cause": effect_name,
                                "effect": str(rel),
                                "depth": d + 1,
                                "strength": 0.5 * (0.8 ** d),
                                "domain": dom.name,
                            })
                            next_frontier.append(str(rel))
            frontier = next_frontier
        return downstream

    def _identify_divergences(self, scenario: str, change: str,
                               direct_effects: List[dict]) -> List[dict]:
        divergences = []
        for i, effect in enumerate(direct_effects[:5]):
            divergences.append({
                "point": f"After '{change[:50]}', {effect['effect'][:50]}",
                "magnitude": effect.get("strength", 0.5),
                "reversibility": 1.0 - effect.get("strength", 0.5),
                "order": i + 1,
            })
        return divergences

    def _assess_counterfactual_plausibility(self, scenario: str, change: str,
                                              direct: List[dict],
                                              downstream: List[dict]) -> float:
        if not direct:
            return 0.3
        connection_score = min(1.0, len(direct) / 3.0)
        avg_strength = sum(e.get("strength", 0) for e in direct) / max(len(direct), 1)
        coherence = 1.0 if len(downstream) > 0 else 0.5
        return round(min(1.0, connection_score * 0.3 + avg_strength * 0.3 + coherence * 0.4), 3)

    def _generate_alternative_outcomes(self, scenario: str, change: str,
                                        downstream: List[dict]) -> List[dict]:
        alternatives = []
        if downstream:
            strongest = max(downstream, key=lambda e: e.get("strength", 0))
            alternatives.append({
                "description": f"Most likely path: leads to '{strongest['effect']}'",
                "probability": strongest.get("strength", 0.5),
                "path": "strongest_causal",
            })
            weakest = min(downstream, key=lambda e: e.get("strength", 0))
            alternatives.append({
                "description": f"Butterfly effect: small change at '{weakest['cause']}' "
                               f"amplifies to '{weakest['effect']}'",
                "probability": weakest.get("strength", 0.3),
                "path": "butterfly",
            })
        return alternatives

    # ── Emergent Insight ────────────────────────────────────────────────

    def emergent_insight(self, concepts: List[str]) -> dict:
        """Combine unrelated concepts to generate novel insights."""
        with self._lock:
            if len(concepts) < 2:
                return {"concepts": concepts, "insight": "Need at least 2 concepts.", "novelty": 0.0}

            connections = self._find_concept_connections(concepts)
            gaps = self._find_connection_gaps(concepts, connections)
            emergent_props = self._identify_emergent_properties(concepts, connections)
            insight_statement = self._synthesize_insight(concepts, connections, emergent_props)
            novelty = self._assess_novelty(concepts, connections)

            result = {
                "concepts": concepts,
                "connections_found": connections,
                "gaps": gaps,
                "emergent_properties": emergent_props,
                "insight": insight_statement,
                "novelty": round(novelty, 3),
                "timestamp": _now(),
            }

            self._insights.append({
                "concepts": concepts,
                "novelty": novelty,
                "insight": insight_statement[:200],
                "timestamp": _now(),
            })
            self._meta["total_insights"] += 1
            self._save()
            return result

    def _find_concept_connections(self, concepts: List[str]) -> List[dict]:
        connections = []
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                if i >= j:
                    continue
                # Check domain co-occurrence
                for domain in self._domains.values():
                    c1_in = c1.lower() in [k.lower() for k in domain.entities]
                    c2_in = c2.lower() in [k.lower() for k in domain.entities]
                    if c1_in and c2_in:
                        connections.append({
                            "concepts": [c1, c2],
                            "type": "co_occurrence",
                            "detail": f"Both in domain '{domain.name}'",
                            "strength": 0.5,
                        })
                # Check analogies
                for analogy in self._analogies:
                    c1_in = c1.lower() in [k.lower() for k in analogy.mappings]
                    c2_in = c2.lower() in [v.lower() for v in analogy.mappings.values()]
                    if c1_in and c2_in:
                        connections.append({
                            "concepts": [c1, c2],
                            "type": "analogical",
                            "detail": f"Connected via analogy: {analogy.base_domain} ↔ {analogy.target_domain}",
                            "strength": analogy.structural_similarity,
                        })
        return connections

    def _find_connection_gaps(self, concepts: List[str],
                               connections: List[dict]) -> List[dict]:
        connected_pairs = set()
        for conn in connections:
            pair = tuple(sorted(conn["concepts"]))
            connected_pairs.add(pair)
        gaps = []
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                if i >= j:
                    continue
                pair = tuple(sorted([c1, c2]))
                if pair not in connected_pairs:
                    gaps.append({
                        "concepts": list(pair),
                        "opportunity": f"Bridge between '{c1}' and '{c2}' could yield novel insight",
                    })
        return gaps

    def _identify_emergent_properties(self, concepts: List[str],
                                       connections: List[dict]) -> List[str]:
        props = []
        domains_involved = set()
        for concept in concepts:
            for domain in self._domains.values():
                if concept.lower() in [k.lower() for k in domain.entities]:
                    domains_involved.add(domain.name)
        if len(domains_involved) > 1:
            props.append(
                f"Cross-domain synthesis: concepts span {', '.join(domains_involved)}"
            )
        if len(concepts) >= 3:
            props.append(f"Ternary blend: {len(concepts)}-way combination increases novelty")
        strong = [c for c in connections if c.get("strength", 0) > 0.7]
        if strong:
            props.append(f"Strong resonance: {len(strong)} high-strength connections")
        return props

    def _synthesize_insight(self, concepts: List[str], connections: List[dict],
                            emergent_props: List[str]) -> str:
        parts = [f"Blending concepts: {', '.join(concepts[:5])}"]
        if connections:
            strongest = max(connections, key=lambda c: c.get("strength", 0))
            parts.append(f"Strongest bridge: {strongest['concepts'][0]} ↔ {strongest['concepts'][1]}")
        if emergent_props:
            parts.append(emergent_props[0][:100])
        return " | ".join(parts)

    def _assess_novelty(self, concepts: List[str], connections: List[dict]) -> float:
        connection_factor = 1.0 - min(1.0, len(connections) / (len(concepts) * (len(concepts) - 1) / 2))
        domains = set()
        for concept in concepts:
            for domain in self._domains.values():
                if concept.lower() in [k.lower() for k in domain.entities]:
                    domains.add(domain.name)
        domain_factor = min(1.0, len(domains) / max(len(concepts), 1))
        return min(1.0, connection_factor * 0.5 + domain_factor * 0.5)

    # ── Cross-Domain Transfer ───────────────────────────────────────────

    def cross_domain_transfer(self, problem: str,
                               known_domains: List[str]) -> List[dict]:
        """Apply solutions from known domains to a novel problem."""
        with self._lock:
            transfers = []
            problem_features = self._extract_problem_features(problem)

            for domain_name in known_domains:
                domain = self._domains.get(domain_name)
                if not domain:
                    continue

                match_score = self._domain_problem_match(problem_features, domain)
                if match_score < STRUCTURAL_SIMILARITY_THRESHOLD:
                    continue

                patterns = self._extract_transferable_patterns(domain, problem_features)
                for pattern in patterns:
                    applicability = match_score * pattern.get("relevance", 0.5)
                    transfers.append({
                        "source_domain": domain_name,
                        "pattern": pattern["description"],
                        "applicability": round(applicability, 3),
                        "adaptation_guidance": self._generate_adaptation_guidance(
                            domain_name, problem, pattern
                        ),
                        "confidence": round(min(1.0, applicability + 0.1), 3),
                    })

            transfers.sort(key=lambda t: t["applicability"], reverse=True)
            self._transfers.extend([{
                "problem": problem[:200],
                "domain": t["source_domain"],
                "applicability": t["applicability"],
                "timestamp": _now(),
            } for t in transfers[:5]])
            self._meta["total_transfers"] += len(transfers)
            self._save()
            return transfers[:10]

    def _extract_problem_features(self, problem: str) -> Dict[str, Any]:
        words = problem.lower().split()
        return {
            "word_count": len(words),
            "unique_words": len(set(words)),
            "has_constraints": any(w in problem.lower() for w in ["limit", "constraint", "must", "cannot", "only"]),
            "has_optimization": any(w in problem.lower() for w in ["maximize", "minimize", "optimize", "best", "fastest"]),
            "has_sequence": any(w in problem.lower() for w in ["then", "after", "before", "first", "next", "step"]),
            "has_tradeoff": any(w in problem.lower() for w in ["but", "however", "tradeoff", "versus", "vs"]),
            "keywords": set(words),
        }

    def _domain_problem_match(self, problem_features: Dict[str, Any],
                               domain: Domain) -> float:
        score = 0.0
        factors = 0
        problem_words = problem_features.get("keywords", set())
        domain_words = set()
        for entity in domain.entities:
            domain_words.update(entity.lower().split())
        if problem_words and domain_words:
            overlap = len(problem_words & domain_words)
            score += overlap / max(len(problem_words), 1)
            factors += 1
        if domain.relations:
            score += min(1.0, len(domain.relations) / 10.0)
            factors += 1
        return score / max(factors, 1)

    def _extract_transferable_patterns(self, domain: Domain,
                                        problem_features: Dict[str, Any]) -> List[dict]:
        patterns = []
        for rel in domain.relations:
            relevance = 0.5
            desc = str(rel).lower()
            problem_words = problem_features.get("keywords", set())
            if any(w in desc for w in problem_words):
                relevance = 0.8
            patterns.append({
                "description": f"In {domain.name}: {rel}",
                "relation": rel.name,
                "relevance": relevance,
            })
        return patterns[:5]

    def _generate_adaptation_guidance(self, source_domain: str, problem: str,
                                       pattern: dict) -> str:
        return (
            f"Adapt '{pattern.get('description', 'pattern')}' from {source_domain} "
            f"by mapping the relational structure to your problem context."
        )

    # ── Abstract Pattern ────────────────────────────────────────────────

    def abstract_pattern(self, instances: List[str]) -> dict:
        """Find common patterns across diverse instances."""
        with self._lock:
            if len(instances) < 2:
                return {"instances": instances, "pattern": "Need at least 2 instances.", "abstraction_level": 0.0}

            instance_features = []
            for inst in instances:
                features = {
                    "words": set(inst.lower().split()),
                    "length": len(inst.split()),
                    "has_numbers": any(c.isdigit() for c in inst),
                    "has_negation": any(w in inst.lower() for w in ["not", "no", "never", "without", "fail"]),
                    "has_question": "?" in inst,
                }
                instance_features.append(features)

            common_words = instance_features[0]["words"].copy()
            for feats in instance_features[1:]:
                common_words &= feats["words"]

            stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                          "have", "has", "had", "do", "does", "did", "will", "would", "could",
                          "should", "may", "might", "can", "shall", "to", "of", "in", "for",
                          "on", "with", "at", "by", "from", "it", "this", "that", "and", "or",
                          "but", "if", "as", "into"}
            structural_words = common_words - stop_words

            common_structural = {
                "has_numbers": all(f["has_numbers"] for f in instance_features),
                "has_negation": all(f["has_negation"] for f in instance_features),
                "has_question": all(f["has_question"] for f in instance_features),
                "avg_length": sum(f["length"] for f in instance_features) / len(instance_features),
            }

            variations = []
            for i, inst in enumerate(instances):
                unique_words = instance_features[i]["words"] - common_words
                if unique_words:
                    variations.append({
                        "instance_index": i,
                        "unique_elements": list(unique_words)[:10],
                        "text": inst[:100],
                    })

            abstraction = 1.0 - (len(structural_words) / max(sum(f["length"] for f in instance_features) / len(instance_features), 1)) if structural_words else 0.8

            parts = []
            if structural_words:
                parts.append(f"Shared vocabulary: {', '.join(list(structural_words)[:8])}")
            if common_structural.get("has_numbers"):
                parts.append("All instances involve numerical elements")
            if common_structural.get("has_negation"):
                parts.append("All instances involve negation/constraints")
            if common_structural.get("has_question"):
                parts.append("All instances are interrogative")

            pattern_desc = " | ".join(parts) if parts else "No strong abstract pattern detected"

            result = {
                "instances": [i[:200] for i in instances],
                "common_structure": list(structural_words)[:20],
                "common_features": common_structural,
                "variations": variations[:5],
                "abstraction_level": round(max(0, abstraction), 3),
                "pattern_description": pattern_desc,
                "timestamp": _now(),
            }

            self._patterns.append({
                "num_instances": len(instances),
                "common_count": len(structural_words),
                "abstraction_level": result["abstraction_level"],
                "timestamp": _now(),
            })
            self._save()
            return result

    # ── Format & Stats ──────────────────────────────────────────────────

    def format_for_prompt(self, max_chars: int = 800) -> str:
        stats = self.get_stats()
        parts = [
            "[ABSTRACTION ENGINE — SME-based analogical reasoning]",
            f"Domains: {stats['domains']} | Analogies: {stats['total_analogies']}",
            f"Insights: {stats['total_insights']} | Transfers: {stats['total_transfers']}",
            f"First-Principles: {stats['total_first_principles']} | Counterfactuals: {stats['total_counterfactuals']}",
        ]
        domain_list = self.list_domains()
        if domain_list:
            parts.append(f"Known domains: {', '.join(domain_list[:8])}")
        result = "\n".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars] + "[...]"
        return result

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "domains": len(self._domains),
                "total_analogies": self._meta.get("total_analogies", 0),
                "total_insights": self._meta.get("total_insights", 0),
                "total_transfers": self._meta.get("total_transfers", 0),
                "total_counterfactuals": self._meta.get("total_counterfactuals", 0),
                "total_first_principles": self._meta.get("total_first_principles", 0),
                "session_operations": 0,
            }


# ── Singleton ───────────────────────────────────────────────────────────────

_abstraction_engine = None
_abstraction_lock = threading.Lock()


def get_abstraction_engine() -> AbstractionEngine:
    """Get singleton AbstractionEngine instance."""
    global _abstraction_engine
    if _abstraction_engine is None:
        with _abstraction_lock:
            if _abstraction_engine is None:
                _abstraction_engine = AbstractionEngine()
    return _abstraction_engine


# ── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ae = get_abstraction_engine()

    # Register test domains
    solar = ae.register_domain(
        "solar_system",
        entities={"sun": {"mass": "large"}, "planet": {"orbits": True}, "gravity": {"force": True}},
        relations=[("ATTRACTS", ["sun", "planet"], 1), ("CAUSES", ["gravity", "ATTRACTS(sun, planet)"], 2)],
    )
    atom = ae.register_domain(
        "atom",
        entities={"nucleus": {"mass": "large"}, "electron": {"orbits": True}, "coulomb": {"force": True}},
        relations=[("ATTRACTS", ["nucleus", "electron"], 1), ("CAUSES", ["coulomb", "ATTRACTS(nucleus, electron)"], 2)],
    )

    # Find analogies
    analogies = ae.find_analogies("solar_system", "atom")
    print(f"Analogies found: {len(analogies)}")
    for a in analogies:
        print(f"\n  Score: {a['structural_similarity']:.2f}")
        print(f"  Mappings: {a['mappings']}")
        print(f"  Inferences: {a['candidate_inferences']}")

    print(f"\nStats: {ae.get_stats()}")
    print(f"Prompt:\n{ae.format_for_prompt()}")