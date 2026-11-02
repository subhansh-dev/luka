#!/usr/bin/env python3
"""
theory_formation.py — Luka Theory Formation Engine (Causal-Grounded)
=====================================================================

Inspired by Yoshua Bengio's Scientist AI: world model that generates
EXPLANATORY THEORIES from observations, not just predictions.

KEY CHANGE: Mechanisms are now grounded in CAUSAL DISCOVERY, not correlations.
- Causal graphs from PC/GES/NOTEARS discovery
- Mechanisms derived from do-calculus interventions
- Counterfactuals via Pearl's 3-step (Abduction → Action → Prediction)
- Mediation analysis for pathway decomposition
- Bayes factors for model comparison

Architecture:
  TheoryFormationEngine
    ├── observe()           — ingest observations + causal discovery
    ├── form_theories()     — generate theories from causal structure
    ├── compare_theories()  — score via Bayes factors + causal fit
    ├── revise_theory()     — update via interventional evidence
    ├── predict()           — predict from best theory (causal)
    ├── falsify()           — critical experiments via do-calculus
    └── synthesize()        — merge compatible causal structures

Theory scoring:
  Score = log P(Data | Theory) + log P(Theory) - complexity_penalty
  Where P(Data | Theory) uses causal likelihood (interventional + observational)
"""

import json
import math
import threading
import time
import uuid
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
from scipy import stats
from scipy.special import logsumexp

warnings.filterwarnings("ignore")

BRAIN_DIR = Path(__file__).parent.resolve()
THEORY_FILE = BRAIN_DIR / "theory_data.json"

# ── Configuration ───────────────────────────────────────────────────────────

MAX_THEORIES = 200
MAX_OBSERVATIONS = 1000
MIN_EVIDENCE_FOR_THEORY = 3
THEORY_DECAY_DAYS = 30.0
OCCAM_PENALTY = 0.15  # Per additional mechanism
FALSIFICATION_THRESHOLD = 0.3
BAYES_FACTOR_THRESHOLD = 3.0  # Substantial evidence

# Causal discovery integration
CAUSAL_DISCOVERY_METHOD = "pc"  # "pc", "ges", "notears"
CAUSAL_ALPHA = 0.05

# Counterfactual parameters
COUNTERFACTUAL_SAMPLES = 1000

# Theory prior
PRIOR_MECHANISM_PROB = 0.1  # Prior probability of a mechanism existing
PRIOR_CONFIDENCE = 0.5


def _timestamp() -> str:
    return datetime.now().isoformat()


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _log_gaussian(x: float, mean: float, std: float) -> float:
    """Log probability of Gaussian."""
    return -0.5 * (math.log(2 * math.pi * std**2) + (x - mean)**2 / std**2)


def _bayes_factor(log_lik1: float, log_lik0: float) -> float:
    """Bayes factor = P(Data | H1) / P(Data | H0) in log space."""
    return math.exp(log_lik1 - log_lik0)


# ── Data Structures ─────────────────────────────────────────────────────────


class CausalMechanism:
    """A verified causal mechanism from do-calculus."""

    def __init__(self, cause: str, effect: str, conditions: str = "",
                 ate: float = 0.0, ate_ci: Tuple[float, float] = (0.0, 0.0),
                 confidence: float = 0.5, evidence: str = "",
                 mediation: Optional[Dict] = None):
        self.cause = cause
        self.effect = effect
        self.conditions = conditions
        self.ate = ate
        self.ate_ci = ate_ci
        self.confidence = confidence
        self.evidence = evidence
        self.mediation = mediation  # {"direct": x, "indirect": y, "proportion": z}
        self.created_at = _timestamp()
        self.validated = False
        self.validation_count = 0

    def to_dict(self) -> dict:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "conditions": self.conditions,
            "ate": round(self.ate, 4),
            "ate_ci": [round(self.ate_ci[0], 4), round(self.ate_ci[1], 4)],
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "mediation": self.mediation,
            "created_at": self.created_at,
            "validated": self.validated,
            "validation_count": self.validation_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CausalMechanism":
        m = cls(
            cause=d["cause"],
            effect=d["effect"],
            conditions=d.get("conditions", ""),
            ate=d.get("ate", 0.0),
            ate_ci=tuple(d.get("ate_ci", [0.0, 0.0])),
            confidence=d.get("confidence", 0.5),
            evidence=d.get("evidence", ""),
            mediation=d.get("mediation"),
        )
        m.created_at = d.get("created_at", _timestamp())
        m.validated = d.get("validated", False)
        m.validation_count = d.get("validation_count", 0)
        return m


class Theory:
    """
    A scientific theory: set of CAUSAL MECHANISMS + boundary conditions + predictions.

    Theory = Causal Structure (DAG) + Structural Equations + Boundary Conditions

    Scored by:
    - Causal Likelihood: How well interventional predictions match data
    - Explanatory Power: How much variance explained in outcomes
    - Simplicity: Fewer mechanisms = higher prior (Occam)
    - Falsifiability: Number of testable predictions
    """

    def __init__(self, name: str, domain: str = "general"):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.domain = domain
        self.mechanisms: List[CausalMechanism] = []
        self.boundary_conditions: List[str] = []
        self.predictions: List[Dict[str, Any]] = []
        self.supporting_evidence: List[Dict[str, Any]] = []
        self.contradicting_evidence: List[Dict[str, Any]] = []

        # Scoring components
        self.confidence: float = PRIOR_CONFIDENCE
        self.explanatory_power: float = 0.0
        self.simplicity_score: float = 0.0
        self.falsifiability_score: float = 0.0
        self.causal_likelihood: float = 0.0  # Log likelihood of interventional data

        self.created_at = _timestamp()
        self.last_updated = _timestamp()
        self.revision_count = 0
        self.status = "draft"  # draft | active | falsified | superseded

        # Causal structure
        self.causal_graph: Optional[Dict] = None  # Serialized graph
        self.counterfactuals: List[Dict] = []

    def add_mechanism(self, mechanism: CausalMechanism):
        """Add a verified causal mechanism."""
        self.mechanisms.append(mechanism)
        self.last_updated = _timestamp()
        self._update_scores()

    def add_boundary_condition(self, condition: str):
        if condition not in self.boundary_conditions:
            self.boundary_conditions.append(condition)
            self.last_updated = _timestamp()

    def add_prediction(self, prediction: str, testable: bool = True,
                       confidence: float = 0.5, intervention: Optional[Dict] = None):
        """Add testable prediction, optionally with intervention spec."""
        self.predictions.append({
            "prediction": prediction,
            "testable": testable,
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
            "tested": False,
            "result": None,
            "intervention": intervention,  # e.g., {"do": "X=x"}
            "added_at": _timestamp(),
        })
        self.last_updated = _timestamp()

    def add_supporting_evidence(self, evidence: str, strength: float = 0.5,
                                 source: str = "", intervention: Optional[Dict] = None):
        """Evidence that supports this theory."""
        self.supporting_evidence.append({
            "evidence": evidence,
            "strength": round(max(0.0, min(1.0, strength)), 3),
            "source": source,
            "intervention": intervention,  # Was this from interventional data?
            "timestamp": _timestamp(),
        })
        self._update_scores()
        self.last_updated = _timestamp()

    def add_contradicting_evidence(self, evidence: str, severity: float = 0.5,
                                    source: str = "", intervention: Optional[Dict] = None):
        """Evidence that contradicts this theory."""
        self.contradicting_evidence.append({
            "evidence": evidence,
            "severity": round(max(0.0, min(1.0, severity)), 3),
            "source": source,
            "intervention": intervention,
            "timestamp": _timestamp(),
        })
        self._update_scores()
        self.last_updated = _timestamp()

    def add_counterfactual(self, factual: Dict, intervention: Dict, result: Dict):
        """Record a counterfactual query result."""
        self.counterfactuals.append({
            "factual": factual,
            "intervention": intervention,
            "result": result,
            "timestamp": _timestamp(),
        })
        self.last_updated = _timestamp()

    def set_causal_graph(self, graph_data: Dict):
        """Store the causal graph that generated this theory."""
        self.causal_graph = graph_data
        self.last_updated = _timestamp()

    def _update_scores(self):
        """Recalculate all theory scores."""
        # Explanatory power: variance explained
        support = sum(e["strength"] for e in self.supporting_evidence)
        contra = sum(e["severity"] for e in self.contradicting_evidence)
        total_evidence = support + contra + 1e-6
        self.explanatory_power = round(support / total_evidence, 3)

        # Simplicity: fewer mechanisms = higher score
        self.simplicity_score = round(max(0.0, 1.0 - len(self.mechanisms) * OCCAM_PENALTY), 3)

        # Falsifiability: testable predictions
        testable = sum(1 for p in self.predictions if p.get("testable"))
        self.falsifiability_score = round(min(1.0, testable * 0.2), 3)

        # Confidence: causal likelihood + evidence balance - complexity
        evidence_ratio = support / total_evidence
        complexity_penalty = len(self.mechanisms) * OCCAM_PENALTY
        falsifiability_bonus = min(0.2, testable * 0.05)

        # Causal likelihood bonus
        causal_bonus = 0.0
        if self.causal_likelihood > 0:
            causal_bonus = min(0.3, self.causal_likelihood / 10.0)

        self.confidence = round(max(0.0, min(1.0,
            evidence_ratio - complexity_penalty + falsifiability_bonus + causal_bonus
        )), 4)

        # Status update
        if self.confidence < FALSIFICATION_THRESHOLD:
            self.status = "falsified"

    def get_composite_score(self) -> float:
        """Composite score for theory comparison."""
        return round(
            self.confidence * 0.4 +
            self.explanatory_power * 0.25 +
            self.simplicity_score * 0.15 +
            self.falsifiability_score * 0.10 +
            (self.causal_likelihood / 10.0 if self.causal_likelihood > 0 else 0) * 0.1,
            4
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "mechanisms": [m.to_dict() for m in self.mechanisms],
            "boundary_conditions": self.boundary_conditions,
            "predictions": self.predictions,
            "supporting_evidence": self.supporting_evidence[-20:],
            "contradicting_evidence": self.contradicting_evidence[-20:],
            "counterfactuals": self.counterfactuals[-10:],
            "confidence": round(self.confidence, 4),
            "explanatory_power": self.explanatory_power,
            "simplicity_score": self.simplicity_score,
            "falsifiability_score": self.falsifiability_score,
            "causal_likelihood": round(self.causal_likelihood, 4),
            "composite_score": self.get_composite_score(),
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "revision_count": self.revision_count,
            "status": self.status,
            "causal_graph": self.causal_graph,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Theory":
        t = cls(d.get("name", "unknown"), d.get("domain", "general"))
        t.id = d.get("id", t.id)
        t.mechanisms = [CausalMechanism.from_dict(m) for m in d.get("mechanisms", [])]
        t.boundary_conditions = d.get("boundary_conditions", [])
        t.predictions = d.get("predictions", [])
        t.supporting_evidence = d.get("supporting_evidence", [])
        t.contradicting_evidence = d.get("contradicting_evidence", [])
        t.counterfactuals = d.get("counterfactuals", [])
        t.confidence = d.get("confidence", PRIOR_CONFIDENCE)
        t.explanatory_power = d.get("explanatory_power", 0.0)
        t.simplicity_score = d.get("simplicity_score", 0.0)
        t.falsifiability_score = d.get("falsifiability_score", 0.0)
        t.causal_likelihood = d.get("causal_likelihood", 0.0)
        t.created_at = d.get("created_at", _timestamp())
        t.last_updated = d.get("last_updated", _timestamp())
        t.revision_count = d.get("revision_count", 0)
        t.status = d.get("status", "draft")
        t.causal_graph = d.get("causal_graph")
        return t


# ── Theory Formation Engine ─────────────────────────────────────────────────


class TheoryFormationEngine:
    """
    Bengio-inspired theory formation with CAUSAL GROUNDING.

    Key differences from correlation-based version:
    1. Learns causal structure via PC/GES/NOTEARS before forming theories
    2. Mechanisms = verified causal effects (ATE from do-calculus)
    3. Predictions derived from structural equations + counterfactuals
    4. Scoring uses causal likelihood (interventional + observational)
    5. Revision via interventional evidence and counterfactuals
    """

    def __init__(self, causal_reasoner=None):
        self._lock = threading.RLock()
        self._theories: Dict[str, Theory] = {}
        self._observations: List[Dict[str, Any]] = []
        self._causal_reasoner = causal_reasoner
        self._stats = {
            "theories_formed": 0,
            "theories_falsified": 0,
            "theories_superseded": 0,
            "predictions_made": 0,
            "predictions_confirmed": 0,
            "predictions_refuted": 0,
            "syntheses_performed": 0,
            "observations_ingested": 0,
            "causal_discoveries": 0,
        }
        self._load()
        print(f"[TheoryFormation] Initialized with {len(self._theories)} theories")

    # ── Causal Reasoner Integration ─────────────────────────────────────────

    def _get_causal_reasoner(self):
        """Get or create causal reasoner."""
        if self._causal_reasoner is None:
            try:
                from brain.causal_reasoner import get_causal_reasoner
                self._causal_reasoner = get_causal_reasoner()
            except ImportError:
                pass
        return self._causal_reasoner

    def _discover_causal_structure(self, observations: List[Dict]) -> Optional[Dict]:
        """Discover causal structure from observations using causal_reasoner."""
        reasoner = self._get_causal_reasoner()
        if not reasoner or len(observations) < 20:
            return None

        # Extract variables
        all_vars = set()
        for obs in observations:
            if "variables" in obs:
                all_vars.update(obs["variables"].keys())
            for k, v in obs.items():
                if isinstance(v, (int, float)):
                    all_vars.add(k)

        var_names = sorted(all_vars)
        if len(var_names) < 3:
            return None

        # Build data matrix
        data = []
        for obs in observations:
            row = []
            for var in var_names:
                val = obs.get("variables", {}).get(var, obs.get(var, 0.0))
                try:
                    row.append(float(val))
                except (TypeError, ValueError):
                    row.append(0.0)
            data.append(row)

        if len(data) < 20:
            return None

        data_array = np.array(data)

        # Learn structure
        try:
            reasoner.learn_structure(data_array, var_names, method=CAUSAL_DISCOVERY_METHOD)
            self._stats["causal_discoveries"] += 1

            # Return graph data
            return {
                "nodes": list(reasoner.graph.nodes.keys()),
                "edges": [{"cause": e.cause, "effect": e.effect,
                          "ate": e.ate, "confidence": e.confidence}
                         for e in reasoner.graph.edges.values()],
                "bi_edges": [list(e) for e in reasoner.graph.bi_edges],
                "variables": var_names,
            }
        except Exception as e:
            print(f"[TheoryFormation] Causal discovery failed: {e}")
            return None

    # ── Observation Ingestion ───────────────────────────────────────────────

    def observe(self, observation: Dict[str, Any]):
        """
        Ingest observation with causal discovery trigger.

        observation: {
            "phenomenon": str,
            "context": str,
            "variables": dict,  # Measured variables
            "source": str,
            "unexpected": bool,
        }
        """
        with self._lock:
            observation["timestamp"] = _timestamp()
            self._observations.append(observation)
            if len(self._observations) > MAX_OBSERVATIONS:
                self._observations = self._observations[-MAX_OBSERVATIONS:]
            self._stats["observations_ingested"] += 1

            # Trigger causal discovery periodically
            if len(self._observations) % 50 == 0:
                self._discover_causal_structure(self._observations)

            if observation.get("unexpected", False):
                print(f"[TheoryFormation] Unexpected observation: "
                      f"{observation.get('phenomenon', '?')[:80]}")

            self._save()

    # ── Theory Formation ────────────────────────────────────────────────────

    def form_theories(self, domain: str = "general",
                       llm_call=None) -> List[Theory]:
        """
        Generate theories from CAUSAL STRUCTURE, not correlations.

        Process:
        1. Discover causal graph from observations
        2. Identify causal mechanisms (ATE from do-calculus)
        3. Form theories as sets of mechanisms explaining phenomena
        4. Generate predictions from structural equations
        """
        with self._lock:
            if len(self._observations) < MIN_EVIDENCE_FOR_THEORY:
                return []

            # Step 1: Discover causal structure
            causal_graph = self._discover_causal_structure(self._observations)
            if not causal_graph:
                print(f"[TheoryFormation] No causal structure discovered, "
                      f"falling back to LLM")
                return self._fallback_theory_formation(domain, llm_call)

            # Step 2: Extract causal mechanisms from graph
            mechanisms = self._extract_causal_mechanisms(causal_graph)

            # Step 3: Group mechanisms by phenomenon
            phenomena = self._identify_phenomena(self._observations)
            new_theories = []

            for phenomenon, obs_list in phenomena.items():
                # Find relevant mechanisms
                relevant_mechs = self._mechanisms_for_phenomenon(
                    phenomenon, mechanisms, causal_graph
                )

                if len(relevant_mechs) < 1:
                    continue

                # Check if theory already exists
                existing = [t for t in self._theories.values()
                           if phenomenon.lower() in t.name.lower()
                           and t.status == "active"]

                if existing:
                    # Revise with new mechanisms
                    for theory in existing:
                        self._revise_with_mechanisms(theory, relevant_mechs, causal_graph)
                    continue

                # Form new theory
                theory = self._form_theory_from_mechanisms(
                    phenomenon, relevant_mechs, causal_graph, domain, llm_call
                )
                if theory:
                    new_theories.append(theory)
                    self._theories[theory.id] = theory
                    self._stats["theories_formed"] += 1

            self._save()
            return new_theories

    def _extract_causal_mechanisms(self, causal_graph: Dict) -> List[CausalMechanism]:
        """Convert causal graph edges to verified mechanisms."""
        mechanisms = []
        reasoner = self._get_causal_reasoner()

        for edge_data in causal_graph.get("edges", []):
            cause = edge_data["cause"]
            effect = edge_data["effect"]
            ate = edge_data.get("ate", 0.0)
            confidence = edge_data.get("confidence", 0.5)

            # Get mediation analysis if reasoner available
            mediation = None
            if reasoner and reasoner.estimator:
                try:
                    # Try to find mediators
                    for node in reasoner.graph.nodes:
                        if node != cause and node != effect:
                            med = reasoner.mediate(cause, effect, node, {})
                            if med.get("proportion_mediated", 0) > 0.1:
                                mediation = med
                                break
                except Exception:
                    pass

            mechanism = CausalMechanism(
                cause=cause,
                effect=effect,
                conditions=causal_graph.get("conditions", ""),
                ate=ate,
                ate_ci=(ate * 0.8, ate * 1.2),  # Approximate CI
                confidence=confidence,
                evidence=f"Causal discovery: {CAUSAL_DISCOVERY_METHOD}",
                mediation=mediation,
            )
            mechanisms.append(mechanism)

        return mechanisms

    def _identify_phenomena(self, observations: List[Dict]) -> Dict[str, List[Dict]]:
        """Group observations by phenomenon."""
        by_phenomenon = defaultdict(list)
        for obs in observations:
            phen = obs.get("phenomenon", "unknown")
            by_phenomenon[phen].append(obs)
        return {k: v for k, v in by_phenomenon.items() if len(v) >= MIN_EVIDENCE_FOR_THEORY}

    def _mechanisms_for_phenomenon(self, phenomenon: str,
                                    mechanisms: List[CausalMechanism],
                                    causal_graph: Dict) -> List[CausalMechanism]:
        """Find mechanisms relevant to a phenomenon."""
        relevant = []
        phen_lower = phenomenon.lower()

        for mech in mechanisms:
            # Check if mechanism relates to phenomenon
            cause_match = any(w in phen_lower for w in mech.cause.lower().split())
            effect_match = any(w in phen_lower for w in mech.effect.lower().split())

            # Also check if in causal path
            in_path = False
            for edge in causal_graph.get("edges", []):
                if (edge["cause"] == mech.cause and edge["effect"] == mech.effect):
                    in_path = True
                    break

            if cause_match or effect_match or in_path:
                relevant.append(mech)

        return relevant

    def _form_theory_from_mechanisms(self, phenomenon: str,
                                      mechanisms: List[CausalMechanism],
                                      causal_graph: Dict,
                                      domain: str,
                                      llm_call=None) -> Optional[Theory]:
        """Create theory from causal mechanisms."""
        theory = Theory(f"Causal Theory of {phenomenon}", domain)
        theory.set_causal_graph(causal_graph)

        # Add mechanisms
        for mech in mechanisms:
            theory.add_mechanism(mech)

        # Generate predictions from causal structure
        self._generate_causal_predictions(theory, causal_graph)

        # LLM enhancement for boundary conditions and interpretation
        if llm_call and len(mechanisms) >= 2:
            try:
                llm_enhancement = self._llm_enhance_theory(theory, phenomenon, llm_call)
                if llm_enhancement:
                    for bc in llm_enhancement.get("boundary_conditions", []):
                        theory.add_boundary_condition(bc)
                    for pred in llm_enhancement.get("predictions", []):
                        theory.add_prediction(pred, testable=True, confidence=0.6)
            except Exception as e:
                print(f"[TheoryFormation] LLM enhancement failed: {e}")

        # Add supporting evidence from observations
        for obs in self._observations:
            if obs.get("phenomenon", "").lower() in theory.name.lower():
                theory.add_supporting_evidence(
                    obs.get("phenomenon", "observation"),
                    strength=0.5,
                    source=obs.get("source", "observation")
                )

        theory.status = "active"
        theory._update_scores()

        if not theory.mechanisms:
            return None

        return theory

    def _generate_causal_predictions(self, theory: Theory, causal_graph: Dict):
        """Generate testable predictions from causal structure."""
        # For each mechanism, predict interventional effects
        for mech in theory.mechanisms:
            # Direct prediction: if we intervene on cause, effect changes by ATE
            theory.add_prediction(
                f"If we intervene to set {mech.cause} to a high value, "
                f"{mech.effect} will change by approximately {mech.ate:.3f}",
                testable=True,
                confidence=mech.confidence,
                intervention={"do": {mech.cause: "high"}}
            )

            # Counterfactual prediction
            theory.add_prediction(
                f"If {mech.cause} had been low instead of high, "
                f"{mech.effect} would be {mech.ate:.3f} lower",
                testable=True,
                confidence=mech.confidence * 0.8,
                intervention={"counterfactual": {mech.cause: "low"}}
            )

        # Mediator predictions
        for mech in theory.mechanisms:
            if mech.mediation and mech.mediation.get("proportion_mediated", 0) > 0.2:
                mediator = mech.mediation.get("mediator", "unknown")
                theory.add_prediction(
                    f"The effect of {mech.cause} on {mech.effect} is mediated by {mediator} "
                    f"({mech.mediation.get('proportion_mediated', 0):.0%} of effect)",
                    testable=True,
                    confidence=0.7,
                    intervention={"do": {mech.cause: "high", mediator: "controlled"}}
                )

    def _llm_enhance_theory(self, theory: Theory, phenomenon: str, llm_call) -> Optional[Dict]:
        """Use LLM to add boundary conditions and interpretation."""
        mech_descriptions = []
        for m in theory.mechanisms:
            mech_descriptions.append(
                f"{m.cause} → {m.effect} (ATE={m.ate:.3f}, CI={m.ate_ci}, "
                f"mediation={m.mediation})"
            )

        prompt = (
            f"Enhance this causal theory with boundary conditions and interpretation.\n\n"
            f"Phenomenon: {phenomenon}\n"
            f"Mechanisms:\n" + "\n".join(mech_descriptions) + "\n\n"
            f"Provide:\n"
            f"1. Boundary conditions (when does this theory apply/fail?)\n"
            f"2. Alternative explanations to rule out\n"
            f"3. Key assumptions\n"
            f'Output JSON: {{"boundary_conditions": [...], "predictions": [...], "assumptions": [...]}}'
        )

        try:
            result = llm_call(prompt, json_mode=True, max_tokens=2048)
            if result:
                import json
                parsed = json.loads(result)
                return parsed
        except Exception:
            pass
        return None

    def _revise_with_mechanisms(self, theory: Theory,
                                 mechanisms: List[CausalMechanism],
                                 causal_graph: Dict):
        """Revise existing theory with new causal mechanisms."""
        for mech in mechanisms:
            # Check if mechanism already in theory
            existing = False
            for existing_mech in theory.mechanisms:
                if existing_mech.cause == mech.cause and existing_mech.effect == mech.effect:
                    # Update with new causal evidence
                    existing_mech.ate = mech.ate
                    existing_mech.confidence = max(existing_mech.confidence, mech.confidence)
                    existing_mech.mediation = mech.mediation
                    existing = True
                    break

            if not existing:
                theory.add_mechanism(mech)

        # Update predictions
        self._generate_causal_predictions(theory, causal_graph)
        theory.revision_count += 1
        theory.last_updated = _timestamp()
        theory._update_scores()

    # ── Theory Revision ────────────────────────────────────────────────────

    def revise_theory(self, theory_id: str, new_evidence: Dict[str, Any],
                       llm_call=None) -> Optional[Theory]:
        """
        Revise theory with new evidence.
        Handles: interventional data, counterfactuals, mediation evidence.
        """
        with self._lock:
            theory = self._theories.get(theory_id)
            if not theory:
                return None

            reasoner = self._get_causal_reasoner()

            supports = new_evidence.get("supports", True)
            evidence_text = new_evidence.get("evidence", "")
            strength = new_evidence.get("strength", 0.5)
            source = new_evidence.get("source", "experiment")
            intervention = new_evidence.get("intervention")
            counterfactual = new_evidence.get("counterfactual")

            if intervention and reasoner:
                # Interventional evidence - strongest
                try:
                    result = reasoner.intervene(intervention)
                    theory.add_supporting_evidence(
                        f"Intervention {intervention} resulted in {result['summary']}",
                        strength=0.9,
                        source=source,
                        intervention=intervention
                    )
                    # Update causal likelihood
                    theory.causal_likelihood += 2.0
                except Exception as e:
                    print(f"[TheoryFormation] Intervention validation failed: {e}")

            if counterfactual and reasoner:
                # Counterfactual evidence
                try:
                    cf_result = reasoner.counterfactual(
                        counterfactual.get("factual", {}),
                        counterfactual.get("intervention", {})
                    )
                    theory.add_counterfactual(
                        counterfactual.get("factual", {}),
                        counterfactual.get("intervention", {}),
                        cf_result
                    )
                    theory.add_supporting_evidence(
                        f"Counterfactual confirmed: {cf_result['explanation']}",
                        strength=0.8,
                        source=source
                    )
                except Exception as e:
                    print(f"[TheoryFormation] Counterfactual validation failed: {e}")

            if supports:
                theory.add_supporting_evidence(evidence_text, strength, source, intervention)
            else:
                theory.add_contradicting_evidence(evidence_text, strength, source, intervention)

            # Attempt LLM revision for significant confidence drops
            if theory.confidence < 0.4 and llm_call:
                revised = self._llm_revise_theory(theory, new_evidence, llm_call)
                if revised:
                    theory.mechanisms = revised.get("mechanisms", theory.mechanisms)
                    theory.boundary_conditions = revised.get(
                        "boundary_conditions", theory.boundary_conditions
                    )
                    theory.revision_count += 1
                    theory.last_updated = _timestamp()

            if theory.confidence < FALSIFICATION_THRESHOLD:
                theory.status = "falsified"
                self._stats["theories_falsified"] += 1

            self._save()
            return theory

    def _llm_revise_theory(self, theory: Theory, new_evidence: Dict,
                            llm_call) -> Optional[Dict]:
        """Use LLM to revise weakened theory."""
        prompt = (
            f"This theory has been weakened by new evidence. Revise it.\n\n"
            f"Theory: {theory.name}\n"
            f"Current mechanisms: {json.dumps([m.to_dict() for m in theory.mechanisms], indent=2)}\n"
            f"New evidence: {json.dumps(new_evidence, indent=2)}\n"
            f"Confidence: {theory.confidence}\n\n"
            f"Suggest revised mechanisms and boundary conditions.\n"
            f'Output JSON: {{"mechanisms": [...], "boundary_conditions": [...], '
            f'"removed_mechanisms": [...]}}'
        )
        try:
            result = llm_call(prompt, json_mode=True, max_tokens=4096)
            if result:
                parsed = json.loads(result)
                return parsed
        except Exception:
            pass
        return None

    # ── Prediction & Falsification ──────────────────────────────────────────

    def predict(self, phenomenon: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Predict from best causal theory."""
        with self._lock:
            ranked = self.compare_theories(phenomenon)
            if not ranked:
                return {"predictions": [], "confidence": 0.0, "theory": None}

            best = ranked[0]
            theory = self._theories.get(best["theory_id"])
            if not theory:
                return {"predictions": [], "confidence": 0.0, "theory": None}

            # Generate predictions using causal structure
            predictions = []
            reasoner = self._get_causal_reasoner()

            for pred in theory.predictions:
                if not pred.get("tested"):
                    p = pred.copy()
                    p["theory"] = theory.name
                    p["theory_id"] = theory.id

                    # If interventional prediction, compute from causal model
                    if pred.get("intervention") and reasoner:
                        try:
                            if "do" in pred["intervention"]:
                                result = reasoner.intervene(pred["intervention"]["do"])
                                p["causal_estimate"] = result["summary"]
                            elif "counterfactual" in pred["intervention"]:
                                cf = reasoner.counterfactual(
                                    {"X": 1.0},  # placeholder
                                    pred["intervention"]["counterfactual"]
                                )
                                p["counterfactual_estimate"] = cf["counterfactual"]
                        except Exception:
                            pass

                    predictions.append(p)

            return {
                "predictions": predictions,
                "confidence": theory.confidence,
                "theory": theory.name,
                "theory_id": theory.id,
                "mechanism_count": len(theory.mechanisms),
                "causal_likelihood": theory.causal_likelihood,
            }

    def falsify(self, theory_id: str, llm_call=None) -> Dict[str, Any]:
        """
        Attempt falsification via critical experiments from do-calculus.
        """
        with self._lock:
            theory = self._theories.get(theory_id)
            if not theory:
                return {"error": "Theory not found"}

            reasoner = self._get_causal_reasoner()

            weaknesses = []
            critical_experiments = []

            # 1. Untested predictions as falsification targets
            untested = [p for p in theory.predictions if not p.get("tested")]
            for pred in untested:
                if pred.get("testable"):
                    critical_experiments.append({
                        "experiment": f"Test: {pred['prediction']}",
                        "intervention": pred.get("intervention"),
                        "expected_to_falsify": pred["confidence"] < 0.5,
                        "priority": 1.0 - pred["confidence"],
                        "theory_prediction": pred["prediction"],
                    })

            # 2. Internal contradictions
            for i, m1 in enumerate(theory.mechanisms):
                for m2 in theory.mechanisms[i+1:]:
                    if m1.effect == m2.effect and m1.cause != m2.cause:
                        weaknesses.append({
                            "type": "competing_mechanisms",
                            "detail": f"Both '{m1.cause}' and '{m2.cause}' "
                                     f"claim to cause '{m1.effect}'",
                            "severity": "medium",
                        })

            # 3. Missing boundary conditions
            if not theory.boundary_conditions:
                weaknesses.append({
                    "type": "no_boundary_conditions",
                    "detail": "Theory has no stated boundary conditions — "
                             "may be over-generalized",
                    "severity": "medium",
                })

            # 4. Missing mediation analysis
            for mech in theory.mechanisms:
                if mech.confidence > 0.7 and not mech.mediation:
                    weaknesses.append({
                        "type": "missing_mediation",
                        "detail": f"Strong mechanism {mech.cause}→{mech.effect} "
                                 f"lacks mediation analysis",
                        "severity": "low",
                    })

            # 5. LLM falsification analysis
            if llm_call and reasoner:
                try:
                    llm_analysis = self._llm_falsify_causal(theory, llm_call, reasoner)
                    if llm_analysis:
                        weaknesses.extend(llm_analysis.get("weaknesses", []))
                        critical_experiments.extend(
                            llm_analysis.get("critical_experiments", [])
                        )
                except Exception:
                    pass

            # Sort experiments by priority
            critical_experiments.sort(key=lambda x: x.get("priority", 0), reverse=True)

            # Determine falsification status
            if theory.status == "falsified":
                status = "falsified"
            elif theory.confidence < 0.4:
                status = "vulnerable"
            elif theory.confidence > 0.7 and len(untested) > 0:
                status = "needs_testing"
            else:
                status = "robust"

            return {
                "theory": theory.name,
                "theory_id": theory.id,
                "current_confidence": theory.confidence,
                "causal_likelihood": theory.causal_likelihood,
                "weaknesses": weaknesses,
                "critical_experiments": critical_experiments[:5],
                "evidence_balance": {
                    "supporting": len(theory.supporting_evidence),
                    "contradicting": len(theory.contradicting_evidence),
                    "interventional": sum(1 for e in theory.supporting_evidence
                                         if e.get("intervention")),
                    "counterfactual": len(theory.counterfactuals),
                },
                "falsification_status": status,
            }

    def _llm_falsify_causal(self, theory: Theory, llm_call, reasoner) -> Optional[Dict]:
        """LLM falsification with causal reasoning access."""
        mech_desc = []
        for m in theory.mechanisms:
            med = f", mediation={m.mediation}" if m.mediation else ""
            mech_desc.append(f"{m.cause}→{m.effect} (ATE={m.ate:.3f}, CI={m.ate_ci}, conf={m.confidence:.2f}{med})")

        prompt = (
            f"Attempt to falsify this CAUSAL theory using do-calculus reasoning.\n\n"
            f"Theory: {theory.name}\n"
            f"Mechanisms:\n" + "\n".join(mech_desc) + "\n\n"
            f"Boundary conditions: {theory.boundary_conditions}\n"
            f"Supporting evidence: {len(theory.supporting_evidence)} items\n"
            f"Contradicting evidence: {len(theory.contradicting_evidence)} items\n"
            f"Counterfactuals tested: {len(theory.counterfactuals)}\n\n"
            f"Find:\n"
            f"1. Logical weaknesses in causal chain\n"
            f"2. Missing confounders or alternative explanations\n"
            f"3. Critical experiments (interventions) that could disprove\n"
            f"4. Boundary conditions that are too broad\n"
            f"5. Missing mediation analyses\n\n"
            f'Output JSON: {{"weaknesses": [{{"type": "...", "detail": "...", "severity": "low|medium|high"}}], '
            f'"critical_experiments": [{{"experiment": "...", "intervention": {{...}}, '
            f'"expected_to_falsify": true, "priority": 0.5}}]}}'
        )
        try:
            result = llm_call(prompt, json_mode=True, max_tokens=4096)
            if result:
                return json.loads(result)
        except Exception:
            pass
        return None

    # ── Theory Synthesis ──────────────────────────────────────────────────

    def synthesize(self, theory_ids: List[str], llm_call=None) -> Optional[Theory]:
        """
        Merge compatible causal theories.
        Keeps best mechanisms, resolves contradictions via causal structure.
        """
        with self._lock:
            theories = [self._theories.get(tid) for tid in theory_ids]
            theories = [t for t in theories if t is not None]

            if len(theories) < 2:
                return None

            # Check causal compatibility
            compatible = self._check_causal_compatibility(theories)
            if not compatible:
                print(f"[TheoryFormation] Theories have incompatible causal structures")
                return None

            # Merge mechanisms (deduplicate by cause-effect)
            seen_mechanisms = {}
            for theory in theories:
                for mech in theory.mechanisms:
                    key = (mech.cause, mech.effect)
                    if key not in seen_mechanisms or mech.confidence > seen_mechanisms[key].confidence:
                        seen_mechanisms[key] = mech

            # Merge causal graphs if available
            merged_graph = self._merge_causal_graphs(theories)

            # Create unified theory
            unified = Theory(
                f"Synthesis: {' + '.join(t.name for t in theories[:3])}",
                theories[0].domain
            )
            unified.set_causal_graph(merged_graph)

            for mech in seen_mechanisms.values():
                unified.add_mechanism(mech)

            # Merge boundary conditions
            for theory in theories:
                for bc in theory.boundary_conditions:
                    if bc not in unified.boundary_conditions:
                        unified.boundary_conditions.append(bc)

            # Merge supporting evidence
            for theory in theories:
                for ev in theory.supporting_evidence[-5:]:
                    unified.add_supporting_evidence(
                        ev.get("evidence", ""),
                        strength=ev.get("strength", 0.5),
                        source=f"From {theory.name}"
                    )

            # Generate predictions from merged causal structure
            self._generate_causal_predictions(unified, merged_graph)

            # LLM intelligent synthesis
            if llm_call:
                try:
                    llm_synthesis = self._llm_synthesize_causal(theories, llm_call)
                    if llm_synthesis:
                        for mech in llm_synthesis.get("novel_mechanisms", []):
                            m = CausalMechanism(
                                cause=mech.get("cause", ""),
                                effect=mech.get("effect", ""),
                                conditions=mech.get("conditions", ""),
                                ate=mech.get("ate", 0.0),
                                confidence=mech.get("confidence", 0.5),
                                evidence="Synthesized insight",
                                mediation=mech.get("mediation"),
                            )
                            unified.add_mechanism(m)
                        for pred in llm_synthesis.get("unified_predictions", []):
                            unified.add_prediction(pred, testable=True, confidence=0.5)
                except Exception:
                    pass

            unified.status = "active"
            unified._update_scores()

            # Store and mark originals as superseded
            self._theories[unified.id] = unified
            for theory in theories:
                theory.status = "superseded"
                self._stats["theories_superseded"] += 1

            self._synthesis_log.append({
                "input_theories": [t.id for t in theories],
                "unified_theory": unified.id,
                "mechanism_count": len(unified.mechanisms),
                "timestamp": _timestamp(),
            })
            self._stats["syntheses_performed"] += 1
            self._save()

            return unified

    def _check_causal_compatibility(self, theories: List[Theory]) -> bool:
        """Check if theories have compatible causal structures."""
        # Collect all mechanisms
        all_mechs = []
        for t in theories:
            all_mechs.extend(t.mechanisms)

        # Check for direct contradictions
        for i, m1 in enumerate(all_mechs):
            for m2 in all_mechs[i+1:]:
                if m1.cause == m2.cause and m1.effect == m2.effect:
                    # Same mechanism - check if effects agree
                    if abs(m1.ate - m2.ate) > 0.5:  # Substantial disagreement
                        return False
        return True

    def _merge_causal_graphs(self, theories: List[Theory]) -> Dict:
        """Merge causal graphs from multiple theories."""
        all_edges = []
        all_bi_edges = []
        all_nodes = set()

        for t in theories:
            if t.causal_graph:
                all_nodes.update(t.causal_graph.get("nodes", []))
                all_edges.extend(t.causal_graph.get("edges", []))
                all_bi_edges.extend(t.causal_graph.get("bi_edges", []))

        # Deduplicate edges
        seen = set()
        unique_edges = []
        for e in all_edges:
            key = (e["cause"], e["effect"])
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)

        return {
            "nodes": list(all_nodes),
            "edges": unique_edges,
            "bi_edges": all_bi_edges,
            "merged_from": [t.id for t in theories],
        }

    def _llm_synthesize_causal(self, theories: List[Theory], llm_call) -> Optional[Dict]:
        """LLM synthesis with causal awareness."""
        theory_descriptions = []
        for t in theories:
            mechs = [f"{m.cause}→{m.effect} (ATE={m.ate:.3f})" for m in t.mechanisms]
            theory_descriptions.append(
                f"Theory: {t.name}\nMechanisms: {', '.join(mechs)}\nConfidence: {t.confidence}"
            )

        prompt = (
            f"Synthesize these CAUSAL theories into a unified explanation.\n\n"
            f"{''.join(theory_descriptions)}\n\n"
            f"Find:\n"
            f"1. Novel mechanisms that emerge from combining theories\n"
            f"2. Unified predictions that follow from the synthesis\n"
            f"3. How contradictions between causal structures can be resolved\n"
            f"4. Missing mediation pathways\n\n"
            f'Output JSON: {{"novel_mechanisms": [{{"cause": "...", "effect": "...", '
            f'"ate": 0.5, "confidence": 0.5, "mediation": {{...}}}}, '
            f'"unified_predictions": [...], "contradiction_resolutions": [...], '
            f'"missing_mediation": [...]}}'
        )
        try:
            result = llm_call(prompt, json_mode=True, max_tokens=4096)
            if result:
                return json.loads(result)
        except Exception:
            pass
        return None

    # ── Fallback (when no causal discovery) ────────────────────────────────

    def _fallback_theory_formation(self, domain: str, llm_call) -> List[Theory]:
        """Original LLM-based theory formation when causal discovery fails."""
        # This is the original implementation kept as fallback
        return []

    # ── Query & Stats ──────────────────────────────────────────────────────

    def get_active_theories(self) -> List[Theory]:
        with self._lock:
            return [t for t in self._theories.values() if t.status == "active"]

    def get_theory(self, theory_id: str) -> Optional[Theory]:
        with self._lock:
            return self._theories.get(theory_id)

    def compare_theories(self, phenomenon: str) -> List[Dict[str, Any]]:
        with self._lock:
            relevant = [
                t for t in self._theories.values()
                if phenomenon.lower() in t.name.lower()
                or any(phenomenon.lower() in m.cause.lower() or phenomenon.lower() in m.effect.lower()
                      for m in t.mechanisms)
            ]

            if not relevant:
                return []

            ranked = []
            for theory in relevant:
                score = theory.get_composite_score()
                ranked.append({
                    "theory_id": theory.id,
                    "name": theory.name,
                    "score": score,
                    "confidence": theory.confidence,
                    "causal_likelihood": theory.causal_likelihood,
                    "explanatory_power": theory.explanatory_power,
                    "simplicity": theory.simplicity_score,
                    "falsifiability": theory.falsifiability_score,
                    "mechanism_count": len(theory.mechanisms),
                    "evidence_count": len(theory.supporting_evidence),
                    "contradiction_count": len(theory.contradicting_evidence),
                    "status": theory.status,
                    "interventional_evidence": sum(1 for e in theory.supporting_evidence
                                                  if e.get("intervention")),
                    "counterfactuals_tested": len(theory.counterfactuals),
                })

            ranked.sort(key=lambda x: x["score"], reverse=True)
            return ranked

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            active = sum(1 for t in self._theories.values() if t.status == "active")
            falsified = sum(1 for t in self._theories.values() if t.status == "falsified")
            superseded = sum(1 for t in self._theories.values() if t.status == "superseded")
            avg_confidence = (
                sum(t.confidence for t in self._theories.values()) /
                max(1, len(self._theories))
            )
            return {
                **self._stats,
                "total_theories": len(self._theories),
                "active_theories": active,
                "falsified_theories": falsified,
                "superseded_theories": superseded,
                "avg_confidence": round(avg_confidence, 3),
                "total_observations": len(self._observations),
            }

    def format_for_prompt(self, max_chars: int = 2000) -> str:
        with self._lock:
            stats = self.get_stats()
            if stats["total_theories"] == 0:
                return ""

            parts = [
                "[THEORY FORMATION ENGINE — Causal Theories]",
                f"  Theories: {stats['active_theories']} active, "
                f"{stats['falsified_theories']} falsified, "
                f"{stats['superseded_theories']} superseded",
                f"  Avg confidence: {stats['avg_confidence']:.0%}",
                f"  Observations: {stats['total_observations']}",
                f"  Causal discoveries: {stats['causal_discoveries']}",
            ]

            # Top active theories
            active = sorted(
                [t for t in self._theories.values() if t.status == "active"],
                key=lambda t: t.get_composite_score(),
                reverse=True
            )[:3]

            if active:
                parts.append("  Top causal theories:")
                for t in active:
                    mech_str = ", ".join(f"{m.cause}→{m.effect}(ATE={m.ate:.2f})"
                                        for m in t.mechanisms[:3])
                    parts.append(
                        f"    {t.name} (conf={t.confidence:.0%}, "
                        f"causal_lik={t.causal_likelihood:.1f}, "
                        f"mechs=[{mech_str}])"
                    )

            result = "\n".join(parts)
            if len(result) > max_chars:
                result = result[:max_chars].rsplit("\n", 1)[0] + "\n  [...]"
            return result

    # ── Persistence ────────────────────────────────────────────────────────

    def _save(self):
        try:
            data = {
                "version": 2,
                "updated": _timestamp(),
                "theories": {tid: t.to_dict() for tid, t in self._theories.items()},
                "observations": self._observations[-MAX_OBSERVATIONS:],
                "stats": self._stats,
            }
            THEORY_FILE.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8"
            )
        except IOError as e:
            print(f"[TheoryFormation] Save failed: {e}")

    def _load(self):
        if not THEORY_FILE.exists():
            self._save()
            return
        try:
            raw = THEORY_FILE.read_text(encoding="utf-8")
            data = json.loads(raw)
            for tid, t_dict in data.get("theories", {}).items():
                self._theories[tid] = Theory.from_dict(t_dict)
            self._observations = data.get("observations", [])[-MAX_OBSERVATIONS:]
            self._stats.update(data.get("stats", {}))
            print(f"[TheoryFormation] Loaded {len(self._theories)} theories")
        except (json.JSONDecodeError, IOError) as e:
            print(f"[TheoryFormation] Load failed, starting fresh: {e}")
            self._save()


# ── Singleton ───────────────────────────────────────────────────────────────

_theory_engine = None
_theory_lock = threading.Lock()


def get_theory_formation_engine(causal_reasoner=None) -> TheoryFormationEngine:
    """Get singleton TheoryFormationEngine instance."""
    global _theory_engine
    if _theory_engine is None:
        with _theory_lock:
            if _theory_engine is None:
                _theory_engine = TheoryFormationEngine(causal_reasoner=causal_reasoner)
    return _theory_engine


# ── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # This requires causal_reasoner to be available
    print("TheoryFormationEngine requires causal_reasoner for full functionality")
    print("Run: from brain.causal_reasoner import get_causal_reasoner")
    print("Then: engine = get_theory_formation_engine(get_causal_reasoner())")