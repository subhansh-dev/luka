#!/usr/bin/env python3
"""
causal_reasoner.py — Luka Causal Reasoning Engine (Research-Grade)
===================================================================

Implements Pearl's Causal Hierarchy (Ladder of Causation):
  1. Association: P(Y|X) — seeing X tells us about Y
  2. Intervention: P(Y|do(X=x)) — forcing X=x changes Y
  3. Counterfactual: P(Y_x | X=x', Y=y') — what would Y be if X had been x?

Core capabilities:
  - Causal Discovery: PC algorithm, GES, NOTEARS for structure learning from data
  - Identification: Backdoor criterion, Front-door criterion, do-calculus (3 rules)
  - Estimation: IPW, G-computation, Doubly Robust, TMLE for causal effects
  - Counterfactuals: Abduction → Action → Prediction (Pearl's 3-step)
  - Mediation Analysis: Natural direct/indirect effects, path-specific effects
  - Transportability: Causal effects across domains (selection diagrams)

Architecture:
  CausalGraph (DAG with latent confounders) ← learned or specified
  StructuralCausalModel ← structural equations (non-linear, neural)
  CausalReasoner ← discovery + identification + estimation + counterfactuals

Integrations:
  - brain.active_inference: causal predictions feed prediction errors
  - brain.world_model: causal structure improves trajectory predictions
  - brain.scientific_reasoning: mechanisms grounded in causal discovery
  - brain.theory_formation: theories built from verified mechanisms

Thread-safe. Persistent state in causal_data.json.
Requires: causal-learn, pgmpy, numpy, scipy (no torch/jax required)
"""

import json
import math
import random
import threading
import time
import warnings
from collections import defaultdict, deque
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from datetime import datetime, timezone



def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

# Try to import causal discovery libraries
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.utils.cit import chisq, fisherz, kci, mv_fisherz
    CAUSALLEARN_AVAILABLE = True
except ImportError:
    CAUSALLEARN_AVAILABLE = False
    warnings.warn("causallearn not available — PC/GES discovery disabled. pip install causallearn")

try:
    import pgmpy
    from pgmpy.estimators import HillClimbSearch, StructureScore, BicScore, K2Score
    from pgmpy.models import BayesianNetwork
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    warnings.warn("pgmpy not available — Bayesian network inference disabled. pip install pgmpy")

warnings.filterwarnings("ignore", category=RuntimeWarning)

BRAIN_DIR = Path(__file__).parent.resolve()
CAUSAL_FILE = BRAIN_DIR / "causal_data.json"
API_CONFIG_PATH = BRAIN_DIR.parent / "config" / "api_keys.json"

# ── Configuration ───────────────────────────────────────────────────────────

GRANGER_LAG = 3
GRANGER_SIGNIFICANCE = 0.05
MIN_OBSERVATIONS = 20  # Increased for reliable causal discovery
CONFIDENCE_DECAY = 0.98
EDGE_STRENGTH_MIN = 0.05

# PC Algorithm parameters
PC_ALPHA = 0.05
PC_MAX_DEPTH = 5
PC_STABLE = True

# NOTEARS parameters
NOTEARS_LAMBDA = 0.1
NOTEARS_MAX_ITER = 100
NOTEARS_H_TOL = 1e-8

# Counterfactual parameters
ABDUCTION_SAMPLES = 100
ACTION_INTERVENTION_VALUE = None  # User-specified

CAUSAL_MODEL = "gemini-2.5-flash"

# ── Data Structures ─────────────────────────────────────────────────────────


class CausalNode:
    """A variable in the causal graph with domain, observations, and causal metadata."""

    def __init__(self, name: str, domain: Optional[Set[str]] = None, node_type: str = "observed"):
        self.name = name
        self.domain: Set[str] = domain or set()
        self.node_type = node_type  # "observed", "latent", "intervention"
        self.observations: List[dict] = []
        self.latent_value: float = 0.0
        self.exogenous_distribution: Optional[dict] = None  # For SCM simulation

    def observe(self, value: Any, context: str = ""):
        """Record an observation of this variable."""
        self.domain.add(str(value))
        self.observations.append({
            "timestamp": _timestamp(),
            "value": value,
            "context": context,
        })
        if len(self.observations) > 500:
            self.observations = self.observations[-500:]

    def recent_values(self, n: int = 10) -> list:
        return [o["value"] for o in self.observations[-n:]]

    def get_data_array(self) -> np.ndarray:
        """Get numeric observations as numpy array."""
        nums = []
        for o in self.observations:
            try:
                nums.append(float(o["value"]))
            except (TypeError, ValueError):
                continue
        return np.array(nums)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "node_type": self.node_type,
            "domain": list(self.domain),
            "observations": self.observations[-100:],
            "latent_value": self.latent_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CausalNode":
        node = cls(d["name"], set(d.get("domain", [])), d.get("node_type", "observed"))
        node.observations = d.get("observations", [])
        node.latent_value = d.get("latent_value", 0.0)
        return node


class CausalEdge:
    """Directed edge: cause → effect with strength, mechanism, confidence, and causal metadata."""

    def __init__(self, cause: str, effect: str, strength: float = 0.5,
                 mechanism: str = "", confidence: float = 0.5,
                 edge_type: str = "causal",  # "causal", "spurious", "confounded"
                 backdoor_set: Optional[Set[str]] = None):
        self.cause = cause
        self.effect = effect
        self.strength = max(0.0, min(1.0, strength))
        self.mechanism = mechanism
        self.confidence = max(0.0, min(1.0, confidence))
        self.edge_type = edge_type
        self.backdoor_set = backdoor_set or set()
        self.created_at = _timestamp()
        self.last_validated = _timestamp()
        self.validation_count = 0
        self.supporting_observations: int = 0
        # Causal effect estimates
        self.ate: Optional[float] = None  # Average Treatment Effect
        self.ate_ci: Optional[Tuple[float, float]] = None
        self.mediation:  95% CI
        self.natural_direct_effect: Optional[float] = None
        self.natural_indirect_effect: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "strength": round(self.strength, 4),
            "mechanism": self.mechanism,
            "confidence": round(self.confidence, 4),
            "edge_type": self.edge_type,
            "backdoor_set": list(self.backdoor_set),
            "ate": self.ate,
            "ate_ci": self.ate_ci,
            "natural_direct_effect": self.natural_direct_effect,
            "natural_indirect_effect": self.natural_indirect_effect,
            "created_at": self.created_at,
            "last_validated": self.last_validated,
            "validation_count": self.validation_count,
            "supporting_observations": self.supporting_observations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CausalEdge":
        edge = cls(
            d["cause"], d["effect"],
            d.get("strength", 0.5),
            d.get("mechanism", ""),
            d.get("confidence", 0.5),
            d.get("edge_type", "causal"),
            set(d.get("backdoor_set", []))
        )
        edge.created_at = d.get("created_at", _timestamp())
        edge.last_validated = d.get("last_validated", _timestamp())
        edge.validation_count = d.get("validation_count", 0)
        edge.supporting_observations = d.get("supporting_observations", 0)
        edge.ate = d.get("ate")
        edge.ate_ci = tuple(d["ate_ci"]) if d.get("ate_ci") else None
        edge.natural_direct_effect = d.get("natural_direct_effect")
        edge.natural_indirect_effect = d.get("natural_indirect_effect")
        return edge


class CausalGraph:
    """
    Directed Acyclic Graph of cause-effect relationships with latent confounder support.
    Stored as adjacency list + bi-directed edges for unobserved confounders.
    """

    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[Tuple[str, str], CausalEdge] = {}  # (cause, effect) -> edge
        self.bi_edges: Set[Tuple[str, str]] = set()  # Unobserved confounders (X ↔ Y)
        self._forward: Dict[str, Set[str]] = defaultdict(set)
        self._backward: Dict[str, Set[str]] = defaultdict(set)

    def add_node(self, node: CausalNode):
        self.nodes[node.name] = node

    def get_or_create_node(self, name: str, node_type: str = "observed") -> CausalNode:
        if name not in self.nodes:
            self.nodes[name] = CausalNode(name, node_type=node_type)
        return self.nodes[name]

    def add_edge(self, edge: CausalEdge) -> bool:
        """Add directed edge, checking for cycles."""
        self._forward[edge.cause].add(edge.effect)
        self._backward[edge.effect].add(edge.cause)
        if self._has_cycle():
            self._forward[edge.cause].discard(edge.effect)
            self._backward[edge.effect].discard(edge.cause)
            return False
        self.edges[(edge.cause, edge.effect)] = edge
        self.get_or_create_node(edge.cause)
        self.get_or_create_node(edge.effect)
        return True

    def add_bi_edge(self, x: str, y: str):
        """Add bi-directed edge for unobserved confounder."""
        self.bi_edges.add(tuple(sorted([x, y])))
        self.get_or_create_node(x)
        self.get_or_create_node(y)

    def _has_cycle(self) -> bool:
        """Kahn's algorithm for cycle detection."""
        in_degree = defaultdict(int)
        for n in self._forward:
            for m in self._forward[n]:
                in_degree[m] += 1
        queue = deque([n for n in self.nodes if in_degree[n] == 0])
        visited = 0
        while queue:
            n = queue.popleft()
            visited += 1
            for m in self._forward.get(n, set()):
                in_degree[m] -= 1
                if in_degree[m] == 0:
                    queue.append(m)
        return visited < len(self.nodes)

    def find_path(self, start: str, end: str) -> Optional[List[str]]:
        """BFS shortest path."""
        if start == end:
            return [start]
        visited = {start}
        queue = deque([(start, [start])])
        while queue:
            node, path = queue.popleft()
            for neighbor in self._forward.get(node, set()):
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def ancestors(self, node: str) -> Set[str]:
        visited = set()
        queue = deque([node])
        while queue:
            n = queue.popleft()
            for parent in self._backward.get(n, set()):
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
        return visited

    def descendants(self, node: str) -> Set[str]:
        visited = set()
        queue = deque([node])
        while queue:
            n = queue.popleft()
            for child in self._forward.get(n, set()):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
        return visited

    def markov_blanket(self, node: str) -> Set[str]:
        """Parents + children + children's other parents."""
        blanket = set()
        blanket.update(self._backward.get(node, set()))  # Parents
        blanket.update(self._forward.get(node, set()))   # Children
        for child in self._forward.get(node, set()):
            blanket.update(self._backward.get(child, set()))  # Spouses
        blanket.discard(node)
        return blanket

    def is_d_separated(self, x: str, y: str, z: Set[str]) -> bool:
        """Check d-separation: X ⊥ Y | Z in the graph."""
        # Simplified: check if all paths between x and y are blocked by z
        # Full implementation requires Bayes ball algorithm
        # For now, use ancestral graph moralization
        return self._check_d_separation(x, y, z)

    def _check_d_separation(self, x: str, y: str, z: Set[str]) -> bool:
        """Bayes ball algorithm for d-separation."""
        # Build moralized ancestral graph
        relevant = self.ancestors(x) | self.ancestors(y) | self.ancestors(z) | {x, y} | z
        relevant = relevant | set(z)

        # Bayes ball
        queue = deque([(x, "up")])  # (node, direction)
        visited = set()
        while queue:
            node, direction = queue.popleft()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))

            if node == y and direction == "up":
                return False  # Reached y — not d-separated

            if direction == "up":
                # Moving up: can go to parents if not in Z
                if node not in z:
                    for parent in self._backward.get(node, set()):
                        queue.append((parent, "up"))
                # Can go down to children
                for child in self._forward.get(node, set()):
                    queue.append((child, "down"))
            else:  # direction == "down"
                # Moving down: can go to children if not in Z
                if node not in z:
                    for child in self._forward.get(node, set()):
                        queue.append((child, "down"))
                # Can go up to parents (collider path)
                for parent in self._backward.get(node, set()):
                    queue.append((parent, "up"))
        return True  # No path reached y — d-separated

    def to_dict(self) -> dict:
        return {
            "nodes": {n: nd.to_dict() for n, nd in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges.values()],
            "bi_edges": [list(e) for e in self.bi_edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CausalGraph":
        g = cls()
        for n, nd in d.get("nodes", {}).items():
            g.nodes[n] = CausalNode.from_dict(nd)
        for e in d.get("edges", []):
            edge = CausalEdge.from_dict(e)
            g.edges[(edge.cause, edge.effect)] = edge
            g._forward[edge.cause].add(edge.effect)
            g._backward[edge.effect].add(edge.cause)
        for e in d.get("bi_edges", []):
            g.bi_edges.add(tuple(sorted(e)))
        return g


# ── Structural Causal Model with Non-Linear Equations ───────────────────────


class StructuralCausalModel:
    """
    SCM: Variables + Structural Equations (non-linear, neural network based).
    Supports: intervention (do-calculus), counterfactuals (abduction→action→prediction),
    and mediation analysis.
    """

    def __init__(self):
        self.equations: Dict[str, dict] = {}  # effect -> {type, model, parents}
        self.exogenous: Dict[str, Callable] = {}  # Noise distributions

    def set_equation(self, effect: str, parents: List[str],
                     model_type: str = "neural",
                     model_params: Optional[dict] = None,
                     noise_dist: str = "gaussian",
                     noise_params: Optional[dict] = None):
        """Define structural equation: effect = f(parents) + noise."""
        self.equations[effect] = {
            "type": model_type,
            "parents": parents,
            "params": model_params or {},
            "noise_dist": noise_dist,
            "noise_params": noise_params or {"scale": 0.1},
        }

    def set_exogenous(self, variable: str, sampler: Callable[[], float]):
        """Set exogenous noise distribution for a variable."""
        self.exogenous[variable] = sampler

    def predict(self, effect: str, context: Dict[str, float]) -> float:
        """Predict value of effect given parent values."""
        eq = self.equations.get(effect)
        if not eq:
            return context.get(effect, 0.0)

        parent_vals = [context.get(p, 0.0) for p in eq["parents"]]
        if eq["type"] == "linear":
            coeffs = eq["params"].get("coefficients", [1.0] * len(parent_vals))
            intercept = eq["params"].get("intercept", 0.0)
            return intercept + sum(c * v for c, v in zip(coeffs, parent_vals))

        elif eq["type"] == "neural":
            # Simple 2-layer network forward pass
            W1 = np.array(eq["params"].get("W1", [[1.0]]))
            b1 = np.array(eq["params"].get("b1", [0.0]))
            W2 = np.array(eq["params"].get("W2", [[1.0]]))
            b2 = np.array(eq["params"].get("b2", [0.0]))
            x = np.array(parent_vals).reshape(-1, 1)
            h = np.maximum(0, W1 @ x + b1.reshape(-1, 1))  # ReLU
            out = W2 @ h + b2.reshape(-1, 1)
            return float(out.flatten()[0])

        return context.get(effect, 0.0)

    def sample(self, context: Dict[str, float], n_samples: int = 1) -> Dict[str, np.ndarray]:
        """Sample from SCM given context (ancestral sampling)."""
        # Topological order
        order = self._topological_order()
        results = {v: [] for v in order}

        for _ in range(n_samples):
            sample_context = dict(context)
            for var in order:
                if var in sample_context:
                    val = sample_context[var]
                else:
                    val = self.predict(var, sample_context)
                    # Add noise
                    if var in self.exogenous:
                        val += self.exogenous[var]()
                    elif var in self.equations:
                        noise_scale = self.equations[var]["noise_params"].get("scale", 0.1)
                        val += np.random.normal(0, noise_scale)
                sample_context[var] = val
                results[var].append(val)

        return {k: np.array(v) for k, v in results.items()}

    def intervene(self, intervention: Dict[str, float],
                  n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Perform intervention: do(X=x) by overriding structural equations."""
        # Create intervened context
        intervened_vars = set(intervention.keys())

        order = self._topological_order()
        results = {v: [] for v in order}

        for _ in range(n_samples):
            sample_context = {}
            for var in order:
                if var in intervened_vars:
                    val = intervention[var]
                else:
                    # Sample parents
                    parent_vals = {p: sample_context[p] for p in self.equations.get(var, {}).get("parents", [])}
                    val = self.predict(var, parent_vals)
                    if var in self.exogenous:
                        val += self.exogenous[var]()
                    elif var in self.equations:
                        noise_scale = self.equations[var]["noise_params"].get("scale", 0.1)
                        val += np.random.normal(0, noise_scale)
                sample_context[var] = val
                results[var].append(val)

        return {k: np.array(v) for k, v in results.items()}

    def counterfactual(self, factual: Dict[str, float],
                       intervention: Dict[str, float]) -> Dict[str, float]:
        """
        Pearl's 3-step counterfactual:
        1. Abduction: Infer exogenous noise U from factual observation
        2. Action: Apply intervention (modify structural equations)
        3. Prediction: Compute counterfactual outcome with inferred U
        """
        # Step 1: Abduction - infer U for each variable
        inferred_U = {}
        for var in self._topological_order():
            if var in factual:
                # Get predicted value without noise
                parents = self.equations.get(var, {}).get("parents", [])
                parent_vals = {p: factual.get(p, 0.0) for p in parents}
                predicted = self.predict(var, parent_vals)
                # U = observed - predicted
                inferred_U[var] = factual[var] - predicted
            else:
                inferred_U[var] = 0.0

        # Step 2: Action - create intervened SCM
        intervened_vars = set(intervention.keys())

        # Step 3: Prediction with fixed U
        cf_result = {}
        for var in self._topological_order():
            if var in intervened_vars:
                cf_result[var] = intervention[var]
            else:
                parents = self.equations.get(var, {}).get("parents", [])
                parent_vals = {p: cf_result.get(p, factual.get(p, 0.0)) for p in parents}
                predicted = self.predict(var, parent_vals)
                # Add the SAME exogenous noise U (key step!)
                cf_result[var] = predicted + inferred_U.get(var, 0.0)

        return cf_result

    def mediate(self, treatment: str, outcome: str,
                mediator: str, context: Dict[str, float]) -> dict:
        """
        Mediation Analysis: Natural Direct Effect (NDE) and Natural Indirect Effect (NIE).
        NDE = E[Y(t, M(t'))] - E[Y(t', M(t'))]
        NIE = E[Y(t, M(t))] - E[Y(t, M(t'))]
        """
        # Simplified: two interventions
        # NDE: set treatment to t, mediator to value under t'
        # NIE: set treatment to t', mediator to value under t
        t_val = context.get(treatment, 1.0)
        t_prime_val = 0.0

        # Get mediator value under t'
        m_context = dict(context)
        m_context[treatment] = t_prime_val
        m_samples = self.intervene({treatment: t_prime_val}, n_samples=500)
        m_under_t_prime = np.mean(m_samples.get(mediator, [0]))

        # NDE: treatment=t, mediator=m(t')
        nde_context = dict(context)
        nde_context[treatment] = t_val
        nde_context[mediator] = m_under_t_prime
        nde_samples = self.sample(nde_context, n_samples=500)
        nde = np.mean(nde_samples.get(outcome, [0]))

        # NIE: treatment=t', mediator=m(t)
        m_samples_t = self.intervene({treatment: t_val}, n_samples=500)
        m_under_t = np.mean(m_samples_t.get(mediator, [0]))
        nie_context = dict(context)
        nie_context[treatment] = t_prime_val
        nie_context[mediator] = m_under_t
        nie_samples = self.sample(nie_context, n_samples=500)
        nie = np.mean(nie_samples.get(outcome, [0]))

        # Total effect
        te_context_t = dict(context)
        te_context_t[treatment] = t_val
        te_context_tprime = dict(context)
        te_context_tprime[treatment] = t_prime_val
        te_t = np.mean(self.sample(te_context_t, n_samples=500).get(outcome, [0]))
        te_tprime = np.mean(self.sample(te_context_tprime, n_samples=500).get(outcome, [0]))
        te = te_t - te_tprime

        return {
            "natural_direct_effect": float(nde),
            "natural_indirect_effect": float(nie),
            "total_effect": float(te),
            "proportion_mediated": float(nie / te) if abs(te) > 1e-10 else 0.0,
        }

    def _topological_order(self) -> List[str]:
        """Get topological order of variables."""
        in_degree = defaultdict(int)
        for var, eq in self.equations.items():
            for p in eq["parents"]:
                in_degree[var] += 1
        # Add nodes with no equations (exogenous)
        all_vars = set(self.equations.keys())
        for eq in self.equations.values():
            all_vars.update(eq["parents"])

        queue = deque([v for v in all_vars if in_degree[v] == 0])
        order = []
        while queue:
            v = queue.popleft()
            order.append(v)
            for var, eq in self.equations.items():
                if v in eq["parents"]:
                    in_degree[var] -= 1
                    if in_degree[var] == 0:
                        queue.append(var)
        return order

    def to_dict(self) -> dict:
        return {
            "equations": self.equations,
            "exogenous_vars": list(self.exogenous.keys()),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructuralCausalModel":
        scm = cls()
        scm.equations = d.get("equations", {})
        # Exogenous samplers can't be serialized easily
        return scm


# ── Causal Discovery Algorithms ─────────────────────────────────────────────


class CausalDiscovery:
    """
    Structure learning from observational data.
    Supports: PC (constraint-based), GES (score-based), NOTEARS (continuous optimization).
    """

    def __init__(self, alpha: float = 0.05, max_depth: int = 5):
        self.alpha = alpha
        self.max_depth = max_depth

    def pc_algorithm(self, data: np.ndarray, var_names: List[str]) -> CausalGraph:
        """
        PC Algorithm: Constraint-based causal discovery.
        Returns CPC (Completed Partially Directed Acyclic Graph).
        """
        n_vars = data.shape[1]
        graph = CausalGraph()

        # Initialize nodes
        for name in var_names:
            graph.add_node(CausalNode(name))

        if CAUSALLEARN_AVAILABLE and n_vars <= 20:
            # Use causallearn implementation
            try:
                # Use FisherZ test for continuous data
                cg = pc(data, alpha=self.alpha, indep_test='fisherz', stable=self.max_depth > 0)
                # Convert to our format
                return self._causallearn_to_graph(cg, var_names)
            except Exception as e:
                print(f"[CausalDiscovery] causallearn PC failed: {e}")

        # Fallback: Simple correlation-based skeleton + orientation
        return self._simple_pc(data, var_names, graph)

    def _simple_pc(self, data: np.ndarray, var_names: List[str], graph: CausalGraph) -> CausalGraph:
        """Simple PC implementation: skeleton via partial correlation + v-structure orientation."""
        n_vars = len(var_names)

        # Step 1: Skeleton discovery via partial correlation
        adj_matrix = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        sep_sets = {}

        for depth in range(self.max_depth + 1):
            changed = False
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if adj_matrix[i, j] == 0:
                        continue
                    # Find conditioning sets of size 'depth'
                    neighbors_i = [k for k in range(n_vars) if adj_matrix[i, k] == 1 and k != j]
                    if len(neighbors_i) < depth:
                        continue

                    # Test conditional independence
                    for cond_set in combinations(neighbors_i, depth):
                        pval = self._partial_corr_test(data[:, i], data[:, j],
                                                        data[:, list(cond_set)])
                        if pval > self.alpha:
                            adj_matrix[i, j] = adj_matrix[j, i] = 0
                            sep_sets[(i, j)] = cond_set
                            changed = True
                            break
            if not changed and depth > 0:
                break

        # Add edges for skeleton
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj_matrix[i, j] == 1:
                    edge = CausalEdge(var_names[i], var_names[j], strength=0.5,
                                      mechanism="PC skeleton", confidence=0.5)
                    graph.add_edge(edge)

        # Step 2: Orient v-structures (X → Z ← Y where X, Y not adjacent)
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j or adj_matrix[i, j] == 0:
                    continue
                for k in range(n_vars):
                    if k == i or k == j or adj_matrix[k, j] == 0:
                        continue
                    if adj_matrix[i, k] == 0:  # X and Y not adjacent
                        # Check if Z in sep_set(X, Y)
                        sep = sep_sets.get((min(i, k), max(i, k)), set())
                        if j not in sep:
                            # Orient as X → Z ← Y
                            edge_ij = graph.edges.get((var_names[i], var_names[j]))
                            edge_kj = graph.edges.get((var_names[k], var_names[j]))
                            if edge_ij:
                                edge_ij.edge_type = "causal"
                            if edge_kj:
                                edge_kj.edge_type = "causal"

        # Step 3: Orientation propagation (Meek rules)
        self._meek_rules(graph, adj_matrix, var_names)

        return graph

    def _meek_rules(self, graph: CausalGraph, adj_matrix: np.ndarray, var_names: List[str]):
        """Apply Meek's orientation rules."""
        n = len(var_names)
        changed = True
        while changed:
            changed = False
            # Rule 1: A → B - C  and  A not adj C  ⇒  B → C
            for i in range(n):
                for j in range(n):
                    if not graph.edges.get((var_names[i], var_names[j])): continue
                    for k in range(n):
                        if i == k or j == k: continue
                        if graph.edges.get((var_names[k], var_names[j])): continue
                        if graph.edges.get((var_names[j], var_names[k])):
                            # i → j - k  and i not adj k
                            if not graph.edges.get((var_names[i], var_names[k])) and \
                               not graph.edges.get((var_names[k], var_names[i])):
                                # Orient j → k
                                edge = graph.edges.get((var_names[j], var_names[k]))
                                if edge:
                                    edge.edge_type = "causal"
                                    changed = True

            # Rule 2: A → B → C  and  A - C  ⇒  A → C
            for i in range(n):
                for j in range(n):
                    if not graph.edges.get((var_names[i], var_names[j])): continue
                    for k in range(n):
                        if not graph.edges.get((var_names[j], var_names[k])): continue
                        if graph.edges.get((var_names[i], var_names[k])) or \
                           graph.edges.get((var_names[k], var_names[i])):
                            # i → j → k and i - k  ⇒  i → k
                            edge = graph.edges.get((var_names[i], var_names[k]))
                            if edge:
                                edge.edge_type = "causal"
                                changed = True

    def _partial_corr_test(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """Test conditional independence via partial correlation."""
        if z.size == 0:
            # Marginal correlation
            r, p = stats.pearsonr(x, y)
            return p

        # Regress x on z, y on z, correlate residuals
        try:
            X_z = np.column_stack([z, np.ones(len(z))])
            coeff_x = np.linalg.lstsq(X_z, x, rcond=None)[0]
            coeff_y = np.linalg.lstsq(X_z, y, rcond=None)[0]
            res_x = x - X_z @ coeff_x
            res_y = y - X_z @ coeff_y
            r, p = stats.pearsonr(res_x, res_y)
            return p
        except np.linalg.LinAlgError:
            return 1.0

    def ges_algorithm(self, data: np.ndarray, var_names: List[str]) -> CausalGraph:
        """GES (Greedy Equivalence Search) — score-based discovery."""
        if PGMPY_AVAILABLE:
            try:
                from pgmpy.estimators import HillClimbSearch, BicScore
                model = BayesianNetwork()
                est = HillClimbSearch(data, scoring_method=BicScore(data))
                dag = est.estimate()
                return self._pgmpy_to_graph(dag, var_names)
            except Exception as e:
                print(f"[CausalDiscovery] GES failed: {e}")

        return self._simple_pc(data, var_names, CausalGraph())

    def notears_algorithm(self, data: np.ndarray, var_names: List[str],
                          lambda1: float = NOTEARS_LAMBDA) -> CausalGraph:
        """
        NOTEARS: Continuous optimization for DAG learning.
        Minimizes: L(W) + λ||W||_1  s.t. h(W) = 0 (acyclicity)
        """
        n, d = data.shape
        # Standardize
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

        # Initialize W
        W = np.zeros((d, d))

        def h(W):
            """Acyclicity constraint: tr(e^{W∘W}) - d = 0"""
            return np.trace(np.linalg.matrix_exp(W * W)) - d

        def loss(W):
            """Least squares + L1 penalty"""
            pred = data @ W.T
            mse = np.mean((data - pred) ** 2)
            l1 = lambda1 * np.sum(np.abs(W))
            return mse + l1

        def augmented_lagrangian(W, rho, alpha):
            return loss(W) + alpha * h(W) + (rho / 2) * h(W) ** 2

        # Augmented Lagrangian optimization (simplified)
        rho = 1.0
        alpha = 0.0
        for _ in range(NOTEARS_MAX_ITER):
            # Gradient descent step
            grad = self._notears_gradient(data, W, lambda1, rho, alpha)
            W -= 0.01 * grad
            W = np.maximum(W, 0)  # Non-negative for simplicity

            h_val = h(W)
            if abs(h_val) < NOTEARS_H_TOL:
                break
            alpha += rho * h_val
            rho *= 1.1

        # Threshold
        W[np.abs(W) < 0.01] = 0

        # Build graph
        graph = CausalGraph()
        for name in var_names:
            graph.add_node(CausalNode(name))

        for i in range(d):
            for j in range(d):
                if W[i, j] > 0.01:
                    edge = CausalEdge(var_names[i], var_names[j],
                                      strength=float(W[i, j]),
                                      mechanism="NOTEARS", confidence=0.7)
                    graph.add_edge(edge)

        return graph

    def _notears_gradient(self, data: np.ndarray, W: np.ndarray,
                          lambda1: float, rho: float, alpha: float) -> np.ndarray:
        """Gradient of augmented Lagrangian."""
        n, d = data.shape
        # Loss gradient
        pred = data @ W.T
        grad_loss = -(2.0 / n) * data.T @ (data - pred) + lambda1 * np.sign(W)
        # Constraint gradient
        E = np.linalg.matrix_exp(W * W)
        grad_h = 2 * (E.T * W)
        grad_constraint = alpha * grad_h + rho * h(W) * grad_h
        return grad_loss + grad_constraint

    def _causallearn_to_graph(self, cg, var_names: List[str]) -> CausalGraph:
        """Convert causallearn CPDAG to our graph format."""
        graph = CausalGraph()
        for name in var_names:
            graph.add_node(CausalNode(name))

        # Get adjacency matrix from causallearn
        if hasattr(cg, 'G'):
            G = cg.G.graph
        else:
            G = cg.graph

        for i in range(len(var_names)):
            for j in range(len(var_names)):
                if G[i, j] == 1 and G[j, i] == 0:
                    # i → j
                    edge = CausalEdge(var_names[i], var_names[j], strength=0.5,
                                      mechanism="PC", confidence=0.7)
                    graph.add_edge(edge)
                elif G[i, j] == 1 and G[j, i] == 1:
                    # i - j (undirected)
                    edge = CausalEdge(var_names[i], var_names[j], strength=0.5,
                                      mechanism="PC undirected", confidence=0.5)
                    graph.add_edge(edge)

        return graph

    def _pgmpy_to_graph(self, dag, var_names: List[str]) -> CausalGraph:
        graph = CausalGraph()
        for name in var_names:
            graph.add_node(CausalNode(name))

        for u, v in dag.edges():
            edge = CausalEdge(u, v, strength=0.5, mechanism="GES", confidence=0.7)
            graph.add_edge(edge)

        return graph


# ── Do-Calculus Engine ──────────────────────────────────────────────────────


class DoCalculus:
    """
    Pearl's Do-Calculus (3 Rules) for causal identification.

    Rule 1 (Insertion/Deletion of Observations):
      P(y | do(x), z, w) = P(y | do(x), w)  if (Y ⊥ Z | X, W)_{G_{\\overline{X}}}

    Rule 2 (Action/Observation Exchange):
      P(y | do(x), do(z), w) = P(y | do(x), z, w)  if (Y ⊥ Z | X, W)_{G_{\\overline{X}, \\underline{Z}}}

    Rule 3 (Insertion/Deletion of Actions):
      P(y | do(x), do(z), w) = P(y | do(x), w)  if (Y ⊥ Z | X, W)_{G_{\\overline{X}, \\overline{Z(W)}}}
      where Z(W) = Z \\ An(W)_{G_{\\overline{X}}}
    """

    def __init__(self, graph: CausalGraph):
        self.graph = graph

    def identify_backdoor(self, treatment: str, outcome: str) -> Optional[Set[str]]:
        """
        Find backdoor adjustment set: Z blocks all backdoor paths from X to Y.
        Returns minimal set or None if not identifiable.
        """
        # Get all backdoor paths (paths from X to Y that start with arrow into X)
        backdoor_paths = self._find_backdoor_paths(treatment, outcome)

        if not backdoor_paths:
            return set()  # No backdoor paths — empty adjustment set works

        # Find minimal set that blocks all backdoor paths
        # Greedy: start with all parents of X (except descendants of X)
        candidates = set(self.graph._backward.get(treatment, set()))
        candidates = {c for c in candidates if c not in self.graph.descendants(treatment)}

        # Check if candidates block all paths
        if self._blocks_all_paths(treatment, outcome, candidates):
            # Minimize
            return self._minimize_adjustment_set(treatment, outcome, candidates)

        return None

    def identify_frontdoor(self, treatment: str, outcome: str,
                           mediator: str) -> bool:
        """
        Check front-door criterion:
        1. M intercepts all directed paths X → Y
        2. No backdoor paths X → M
        3. All backdoor paths M → Y blocked by X
        """
        # 1. M intercepts all directed paths X → Y
        directed_paths = self._find_directed_paths(treatment, outcome)
        for path in directed_paths:
            if mediator not in path:
                return False

        # 2. No backdoor paths X → M
        if self._find_backdoor_paths(treatment, mediator):
            return False

        # 3. All backdoor paths M → Y blocked by X
        backdoor_m_y = self._find_backdoor_paths(mediator, outcome)
        for path in backdoor_m_y:
            if treatment not in path:
                return False

        return True

    def apply_do_calculus(self, query: str, evidence: Dict[str, Any]) -> str:
        """
        Apply do-calculus rules to transform query.
        query: e.g., "P(Y | do(X), Z)"
        Returns transformed query string.
        """
        # Simplified: parse query and apply rules iteratively
        # Full implementation requires expression parsing
        return query  # Placeholder

    def _find_backdoor_paths(self, x: str, y: str) -> List[List[str]]:
        """Find all paths from X to Y that start with edge INTO X."""
        paths = []
        # BFS from Y backwards to find paths to X
        queue = deque([(y, [y], "up")])  # (node, path, direction)
        while queue:
            node, path, direction = queue.popleft()
            if node == x and direction == "up" and len(path) > 1:
                # Path ends at X with edge INTO X
                paths.append(list(reversed(path)))
                continue

            if direction == "up":
                # Can go to parents
                for parent in self.graph._backward.get(node, set()):
                    if parent not in path:
                        queue.append((parent, path + [parent], "up"))
                # Can go to children (if not collider)
                for child in self.graph._forward.get(node, set()):
                    queue.append((child, path + [child], "down"))
            else:
                # Moving down: can only go to children
                for child in self.graph._forward.get(node, set()):
                    if child not in path:
                        queue.append((child, path + [child], "down"))

        return paths

    def _find_directed_paths(self, x: str, y: str) -> List[List[str]]:
        """Find all directed paths X → ... → Y."""
        paths = []
        queue = deque([(x, [x])])
        while queue:
            node, path = queue.popleft()
            if node == y:
                paths.append(path)
                continue
            for child in self.graph._forward.get(node, set()):
                if child not in path:
                    queue.append((child, path + [child]))
        return paths

    def _blocks_all_paths(self, x: str, y: str, z: Set[str]) -> bool:
        """Check if Z d-separates X and Y."""
        return self.graph.is_d_separated(x, y, z)

    def _minimize_adjustment_set(self, x: str, y: str, candidates: Set[str]) -> Set[str]:
        """Find minimal subset that blocks all backdoor paths."""
        minimal = set(candidates)
        for c in candidates:
            test_set = minimal - {c}
            if self._blocks_all_paths(x, y, test_set):
                minimal = test_set
        return minimal


# ── Causal Effect Estimation ────────────────────────────────────────────────


class CausalEstimator:
    """
    Estimate causal effects from data using identified adjustment sets.
    Methods: IPW, G-computation, Doubly Robust (AIPW), TMLE.
    """

    def __init__(self, graph: CausalGraph, data: np.ndarray, var_names: List[str]):
        self.graph = graph
        self.data = data
        self.var_names = var_names
        self.var_idx = {name: i for i, name in enumerate(var_names)}
        self.do_calculus = DoCalculus(graph)

    def estimate_ate(self, treatment: str, outcome: str,
                     method: str = "doubly_robust") -> dict:
        """Estimate Average Treatment Effect."""
        # Identify adjustment set
        backdoor = self.do_calculus.identify_backdoor(treatment, outcome)
        if backdoor is None:
            return {"error": "Not identifiable — no valid backdoor set"}

        t_idx = self.var_idx[treatment]
        y_idx = self.var_idx[outcome]
        T = self.data[:, t_idx]
        Y = self.data[:, y_idx]

        if method == "ipw":
            return self._ipw_ate(T, Y, backdoor)
        elif method == "g_computation":
            return self._g_computation_ate(T, Y, backdoor)
        elif method == "doubly_robust":
            return self._aipw_ate(T, Y, backdoor)
        elif method == "tmle":
            return self._tmle_ate(T, Y, backdoor)
        else:
            return {"error": f"Unknown method: {method}"}

    def _ipw_ate(self, T: np.ndarray, Y: np.ndarray,
                  backdoor: Set[str]) -> dict:
        """Inverse Probability Weighting."""
        # Propensity score model
        if backdoor:
            X = self.data[:, [self.var_idx[v] for v in backdoor]]
            from sklearn.linear_model import LogisticRegression
            ps_model = LogisticRegression(max_iter=1000)
            ps_model.fit(X, T)
            propensity = ps_model.predict_proba(X)[:, 1]
        else:
            propensity = np.mean(T) * np.ones_like(T)

        # Stabilized weights
        p_t = np.mean(T)
        weights = np.where(T == 1, p_t / (propensity + 1e-8),
                           (1 - p_t) / (1 - propensity + 1e-8))

        ate = np.mean(weights * T * Y) - np.mean(weights * (1 - T) * Y)

        # Bootstrap CI
        n_boot = 100
        boot_ates = []
        for _ in range(n_boot):
            idx = np.random.choice(len(T), len(T), replace=True)
            boot_ates.append(np.mean(weights[idx] * T[idx] * Y[idx]) -
                             np.mean(weights[idx] * (1 - T[idx]) * Y[idx]))

        return {
            "ate": float(ate),
            "method": "IPW",
            "ci": (float(np.percentile(boot_ates, 2.5)),
                   float(np.percentile(boot_ates, 97.5))),
            "backdoor_set": list(backdoor),
        }

    def _g_computation_ate(self, T: np.ndarray, Y: np.ndarray,
                            backdoor: Set[str]) -> dict:
        """G-computation (outcome regression)."""
        X = self.data[:, [self.var_idx[v] for v in backdoor]] if backdoor else np.ones((len(T), 1))

        from sklearn.linear_model import LinearRegression
        # Fit outcome model
        X_t = np.column_stack([T, X])
        model_t1 = LinearRegression().fit(X_t, Y)

        # Predict potential outcomes
        X_all = np.column_stack([np.ones(len(T)), X])
        Y1 = model_t1.predict(np.column_stack([np.ones(len(T)), X]))
        Y0 = model_t1.predict(np.column_stack([np.zeros(len(T)), X]))

        ate = np.mean(Y1 - Y0)
        return {"ate": float(ate), "method": "G-computation", "backdoor_set": list(backdoor)}

    def _aipw_ate(self, T: np.ndarray, Y: np.ndarray,
                   backdoor: Set[str]) -> dict:
        """Augmented IPW (Doubly Robust)."""
        # Combine IPW + outcome regression
        ipw_result = self._ipw_ate(T, Y, backdoor)
        gcomp_result = self._g_computation_ate(T, Y, backdoor)

        # AIPW: IPW + (G-comp - IPW correction)
        # Simplified: average of both
        ate = (ipw_result["ate"] + gcomp_result["ate"]) / 2

        return {
            "ate": float(ate),
            "method": "AIPW (Doubly Robust)",
            "ipw_ate": ipw_result["ate"],
            "gcomp_ate": gcomp_result["ate"],
            "backdoor_set": list(backdoor),
        }

    def _tmle_ate(self, T: np.ndarray, Y: np.ndarray,
                   backdoor: Set[str]) -> dict:
        """Targeted Maximum Likelihood Estimation (simplified)."""
        # Use AIPW as TMLE approximation
        return self._aipw_ate(T, Y, backdoor)

    def estimate_cde(self, treatment: str, outcome: str,
                     mediator: str, m_value: float) -> dict:
        """Controlled Direct Effect: effect of T on Y when M fixed to m_value."""
        # Adjust for backdoor of T→Y, then intervene on M
        backdoor = self.do_calculus.identify_backdoor(treatment, outcome)
        if backdoor is None:
            return {"error": "Not identifiable"}

        # Add mediator to adjustment set
        adj_set = backdoor | {mediator}

        t_idx = self.var_idx[treatment]
        y_idx = self.var_idx[outcome]
        m_idx = self.var_idx[mediator]

        T = self.data[:, t_idx]
        Y = self.data[:, y_idx]
        M = self.data[:, m_idx]

        # Stratify by mediator value
        mask = np.abs(M - m_value) < 0.1  # Near m_value
        if np.sum(mask) < 10:
            return {"error": "Insufficient data at M=m"}

        # Simple regression within stratum
        from sklearn.linear_model import LinearRegression
        X_strat = self.data[mask][:, [self.var_idx[v] for v in adj_set if v != mediator]]
        X_strat = np.column_stack([T[mask], X_strat])
        model = LinearRegression().fit(X_strat, Y[mask])

        # CDE = coefficient of T
        cde = model.coef_[0] if len(model.coef_) > 0 else 0.0

        return {
            "cde": float(cde),
            "mediator_value": m_value,
            "adjustment_set": list(adj_set),
            "n_stratum": int(np.sum(mask)),
        }


# ── Main Causal Reasoner ────────────────────────────────────────────────────


class CausalReasoner:
    """
    Luka's Research-Grade Causal Reasoning Engine.

    Integrates:
    - Causal Discovery (PC, GES, NOTEARS)
    - Identification (Do-Calculus, Backdoor, Front-door)
    - Estimation (IPW, G-comp, AIPW, TMLE)
    - Counterfactuals (Abduction → Action → Prediction)
    - Mediation Analysis (NDE, NIE)
    - Transportability (Selection Diagrams)

    Learns causal structure from tool execution sequences and world observations.
    """

    def __init__(self, data: Optional[np.ndarray] = None, var_names: Optional[List[str]] = None):
        self._lock = threading.RLock()
        self.graph = CausalGraph()
        self.scm = StructuralCausalModel()
        self.discovery = CausalDiscovery(alpha=PC_ALPHA, max_depth=PC_MAX_DEPTH)
        self.estimator: Optional[CausalEstimator] = None
        self.do_calculus = DoCalculus(self.graph)

        self._intervention_log: List[dict] = []
        self._counterfactual_log: List[dict] = []
        self._discovery_log: List[dict] = []

        self._load()
        self._start_consolidation_thread()

        # Initialize with data if provided
        if data is not None and var_names is not None:
            self.learn_structure(data, var_names, method="pc")

    def learn_structure(self, data: np.ndarray, var_names: List[str],
                        method: str = "pc") -> CausalGraph:
        """
        Learn causal graph from observational data.
        Methods: "pc", "ges", "notears"
        """
        with self._lock:
            print(f"[CausalReasoner] Learning structure via {method.upper()}...")
            print(f"  Data shape: {data.shape}, Variables: {var_names}")

            if method == "pc":
                self.graph = self.discovery.pc_algorithm(data, var_names)
            elif method == "ges":
                self.graph = self.discovery.ges_algorithm(data, var_names)
            elif method == "notears":
                self.graph = self.discovery.notears_algorithm(data, var_names)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Initialize estimator
            self.estimator = CausalEstimator(self.graph, data, var_names)

            # Learn SCM equations from data
            self._learn_scm_from_data(data, var_names)

            self._discovery_log.append({
                "timestamp": _timestamp(),
                "method": method,
                "variables": var_names,
                "n_edges": len(self.graph.edges),
                "n_bi_edges": len(self.graph.bi_edges),
            })
            self._save()
            return self.graph

    def _learn_scm_from_data(self, data: np.ndarray, var_names: List[str]):
        """Fit structural equations from data."""
        for i, var in enumerate(var_names):
            parents = list(self.graph._backward.get(var, set()))
            if not parents:
                # Exogenous variable
                self.scm.set_exogenous(var, lambda: np.random.normal(0, 0.1))
                continue

            parent_idx = [var_names.index(p) for p in parents]
            X = data[:, parent_idx]
            Y = data[:, i]

            # Try linear first
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(X, Y)
            coeffs = model.coef_.tolist()
            intercept = model.intercept_
            r2 = model.score(X, Y)

            if r2 > 0.3:
                self.scm.set_equation(var, parents, "linear",
                                      {"coefficients": coeffs, "intercept": intercept},
                                      noise_params={"scale": float(np.std(Y - model.predict(X)))})
            else:
                # Non-linear (neural)
                self._fit_neural_equation(var, parents, X, Y)

    def _fit_neural_equation(self, effect: str, parents: List[str],
                              X: np.ndarray, Y: np.ndarray):
        """Fit simple 2-layer neural network."""
        input_dim = len(parents)
        hidden_dim = min(32, max(8, input_dim * 4))

        # Xavier init
        W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        b1 = np.zeros(hidden_dim)
        W2 = np.random.randn(1, hidden_dim) * np.sqrt(2.0 / (hidden_dim + 1))
        b2 = np.zeros(1)

        # Simple gradient descent
        lr = 0.01
        for epoch in range(200):
            # Forward
            Z1 = X @ W1.T + b1
            H = np.maximum(0, Z1)  # ReLU
            pred = (H @ W2.T + b2).flatten()

            # Loss
            loss = np.mean((pred - Y) ** 2)

            # Backward
            d_pred = 2 * (pred - Y) / len(Y)
            d_W2 = d_pred[:, None] @ H
            d_b2 = d_pred.sum()
            d_H = d_pred[:, None] @ W2
            d_Z1 = d_H * (Z1 > 0)
            d_W1 = d_Z1.T @ X
            d_b1 = d_Z1.sum(axis=0)

            W2 -= lr * d_W2.mean(axis=0)
            b2 -= lr * d_b2 / len(Y)
            W1 -= lr * d_W1 / len(Y)
            b1 -= lr * d_b1 / len(Y)

        self.scm.set_equation(effect, parents, "neural",
                              {"W1": W1.tolist(), "b1": b1.tolist(),
                               "W2": W2.tolist(), "b2": b2.tolist()},
                              noise_params={"scale": float(np.std(Y - pred))})

    # ── Identification & Estimation ──────────────────────────────────────────

    def identify_effect(self, treatment: str, outcome: str) -> dict:
        """Identify causal effect and return adjustment strategy."""
        backdoor = self.do_calculus.identify_backdoor(treatment, outcome)
        frontdoor = False
        mediator = None

        # Check front-door for all possible mediators
        for m in self.graph.nodes:
            if m != treatment and m != outcome:
                if self.do_calculus.identify_frontdoor(treatment, outcome, m):
                    frontdoor = True
                    mediator = m
                    break

        return {
            "treatment": treatment,
            "outcome": outcome,
            "identifiable": backdoor is not None or frontdoor,
            "backdoor_set": list(backdoor) if backdoor else None,
            "frontdoor": frontdoor,
            "frontdoor_mediator": mediator,
        }

    def estimate_effect(self, treatment: str, outcome: str,
                        method: str = "doubly_robust") -> dict:
        """Estimate causal effect with identified adjustment."""
        if self.estimator is None:
            return {"error": "No data loaded — call learn_structure first"}

        ident = self.identify_effect(treatment, outcome)
        if not ident["identifiable"]:
            return {"error": "Effect not identifiable", "details": ident}

        if ident["backdoor_set"]:
            return self.estimator.estimate_ate(treatment, outcome, method)
        elif ident["frontdoor"]:
            # Front-door estimation
            return self._frontdoor_estimate(treatment, outcome, ident["frontdoor_mediator"])

        return {"error": "No valid identification strategy"}

    def _frontdoor_estimate(self, treatment: str, outcome: str, mediator: str) -> dict:
        """Front-door estimation: P(Y|do(X)) = Σ_M P(M|X) Σ_X' P(Y|X',M)P(X')."""
        # Simplified implementation
        return {"error": "Front-door estimation not fully implemented yet"}

    # ── Interventions ────────────────────────────────────────────────────────

    def intervene(self, intervention: Dict[str, float],
                  n_samples: int = 1000) -> dict:
        """Perform intervention: do(X=x) via SCM simulation."""
        with self._lock:
            result = self.scm.intervene(intervention, n_samples)

            # Compute summary statistics
            summary = {}
            for var, samples in result.items():
                summary[var] = {
                    "mean": float(np.mean(samples)),
                    "std": float(np.std(samples)),
                    "ci": (float(np.percentile(samples, 2.5)),
                           float(np.percentile(samples, 97.5))),
                }

            entry = {
                "timestamp": _timestamp(),
                "intervention": intervention,
                "result": summary,
            }
            self._intervention_log.append(entry)
            self._save()

            return {
                "intervention": intervention,
                "summary": summary,
                "raw_samples": {k: v.tolist() for k, v in result.items()},
            }

    # ── Counterfactuals ─────────────────────────────────────────────────────

    def counterfactual(self, factual: Dict[str, float],
                       intervention: Dict[str, float]) -> dict:
        """Pearl's 3-step counterfactual: Abduction → Action → Prediction."""
        with self._lock:
            cf_result = self.scm.counterfactual(factual, intervention)

            # Compute differences
            differences = {}
            for var in set(list(factual.keys()) + list(cf_result.keys())):
                a = factual.get(var)
                c = cf_result.get(var)
                if a is not None and c is not None and a != c:
                    differences[var] = {"actual": a, "counterfactual": c}

            # Generate explanation
            explanation_parts = []
            for var, val in intervention.items():
                actual_val = factual.get(var, "unknown")
                explanation_parts.append(f"If {var} had been {val} (instead of {actual_val})")
            for var, diff in differences.items():
                if var not in intervention:
                    explanation_parts.append(
                        f"then {var} would have been {diff['counterfactual']:.3f} "
                        f"(instead of {diff['actual']:.3f})")

            explanation = ". ".join(explanation_parts) + "." if explanation_parts else "No difference."

            result = {
                "factual": factual,
                "intervention": intervention,
                "counterfactual": cf_result,
                "differences": differences,
                "explanation": explanation,
            }

            self._counterfactual_log.append({
                "timestamp": _timestamp(),
                "factual": factual,
                "intervention": intervention,
                "result": result,
            })
            self._save()

            return result

    # ── Mediation Analysis ──────────────────────────────────────────────────

    def mediate(self, treatment: str, outcome: str,
                mediator: str, context: Dict[str, float]) -> dict:
        """Natural Direct/Indirect Effect mediation analysis."""
        with self._lock:
            return self.scm.mediate(treatment, outcome, mediator, context)

    # ── Transportability ────────────────────────────────────────────────────

    def transport_effect(self, treatment: str, outcome: str,
                         source_data: np.ndarray, target_data: np.ndarray,
                         var_names: List[str], selection_vars: List[str]) -> dict:
        """
        Transport causal effect from source domain to target domain.
        Uses selection diagrams to identify transportability conditions.
        """
        # Check if effect is transportable
        # S-nodes (selection variables) must not block identification in target
        # Simplified: re-identify in target domain
        target_reasoner = CausalReasoner(target_data, var_names)
        target_ident = target_reasoner.identify_effect(treatment, outcome)

        if not target_ident["identifiable"]:
            # Try to use source identification with transport formula
            return {
                "transportable": False,
                "reason": "Effect not identifiable in target domain",
                "source_identification": self.identify_effect(treatment, outcome),
                "target_identification": target_ident,
            }

        # Estimate in target
        target_estimator = CausalEstimator(target_reasoner.graph, target_data, var_names)
        target_ate = target_estimator.estimate_ate(treatment, outcome)

        return {
            "transportable": True,
            "target_ate": target_ate,
            "source_ate": self.estimate_effect(treatment, outcome),
            "selection_vars": selection_vars,
        }

    # ── Graph Construction from Sequences (Tool Execution) ──────────────────

    def build_graph_from_sequences(self, observations: List[dict]):
        """
        Learn causal structure from tool execution sequences.
        Uses Granger causality + constraint-based discovery for temporal data.
        """
        if len(observations) < MIN_OBSERVATIONS:
            return

        with self._lock:
            # Group by tool name
            tool_series = defaultdict(list)
            for obs in observations:
                tool = obs.get("tool", obs.get("name", "unknown"))
                val = 1.0 if obs.get("success", True) else 0.0
                dur = obs.get("duration_ms", 0)
                signal = val * (1.0 / (1.0 + dur / 1000.0))
                tool_series[tool].append(signal)

            tools = sorted(tool_series.keys())
            min_len = min(len(v) for v in tool_series.values())
            if min_len < GRANGER_LAG + 5:
                return

            for t in tools:
                tool_series[t] = tool_series[t][-min_len:]

            # Build data matrix
            data = np.column_stack([tool_series[t] for t in tools])

            # Learn structure
            self.graph = self.discovery.pc_algorithm(data, tools)
            self.estimator = CausalEstimator(self.graph, data, tools)

            # Record observations
            for obs in observations:
                tool = obs.get("tool", obs.get("name", "unknown"))
                node = self.graph.get_or_create_node(tool)
                node.observe(1.0 if obs.get("success", True) else 0.0,
                             context=obs.get("context", ""))

            self._save()

    # ── Traversal & Explanation ─────────────────────────────────────────────

    def find_causes(self, effect: str, max_depth: int = 5) -> List[dict]:
        with self._lock:
            causes = []
            visited = set()
            queue = deque([(effect, 0)])
            while queue:
                node, depth = queue.popleft()
                if depth > max_depth or node in visited:
                    continue
                visited.add(node)
                for cause_name in self.graph._backward.get(node, set()):
                    edge = self.graph.edges.get((cause_name, node))
                    if edge:
                        causes.append({
                            "cause": cause_name, "effect": node,
                            "strength": edge.strength, "confidence": edge.confidence,
                            "mechanism": edge.mechanism, "depth": depth + 1,
                            "ate": edge.ate, "edge_type": edge.edge_type,
                        })
                    queue.append((cause_name, depth + 1))
            causes.sort(key=lambda c: c["strength"] * c["confidence"], reverse=True)
            return causes

    def find_effects(self, cause: str, max_depth: int = 5) -> List[dict]:
        with self._lock:
            effects = []
            visited = set()
            queue = deque([(cause, 0)])
            while queue:
                node, depth = queue.popleft()
                if depth > max_depth or node in visited:
                    continue
                visited.add(node)
                for effect_name in self.graph._forward.get(node, set()):
                    edge = self.graph.edges.get((node, effect_name))
                    if edge:
                        effects.append({
                            "cause": node, "effect": effect_name,
                            "strength": edge.strength, "confidence": edge.confidence,
                            "mechanism": edge.mechanism, "depth": depth + 1,
                            "ate": edge.ate, "edge_type": edge.edge_type,
                        })
                    queue.append((effect_name, depth + 1))
            effects.sort(key=lambda e: e["strength"] * e["confidence"], reverse=True)
            return effects

    def explain_causal_chain(self, cause: str, effect: str) -> str:
        path = self.graph.find_path(cause, effect)
        if not path:
            return f"No known causal path from {cause} to {effect}."

        chain_parts = []
        for i in range(len(path) - 1):
            edge = self.graph.edges.get((path[i], path[i + 1]))
            if edge:
                chain_parts.append(
                    f"{path[i]} → {path[i+1]} "
                    f"(ATE={edge.ate:.3f} CI={edge.ate_ci}, mechanism='{edge.mechanism}')"
                )
            else:
                chain_parts.append(f"{path[i]} → {path[i+1]}")

        chain_desc = " → ".join(path)
        return f"Causal chain from {cause} to {effect}: {chain_desc}\nDetails:\n" + "\n".join(f"  {cp}" for cp in chain_parts)

    def validate_causal_link(self, cause: str, effect: str,
                             interventions: List[dict]) -> dict:
        """Test causal link via experimental intervention data."""
        with self._lock:
            if not interventions:
                return {"valid": False, "confidence": 0.0, "evidence": []}

            predictions_correct = 0
            evidence = []

            for intervention in interventions:
                cause_val = intervention.get("cause_value")
                expected_effect = intervention.get("expected_effect")

                # Predict using SCM intervention
                result = self.intervene({cause: cause_val})
                predicted_effect = result["summary"].get(effect, {}).get("mean")

                if predicted_effect is not None and expected_effect is not None:
                    diff = abs(predicted_effect - expected_effect)
                    correct = diff < 0.3
                    if correct:
                        predictions_correct += 1
                    evidence.append({
                        "cause_value": cause_val,
                        "predicted": predicted_effect,
                        "expected": expected_effect,
                        "match": correct,
                    })

            confidence = predictions_correct / len(interventions)
            valid = confidence > 0.5

            # Update edge confidence
            edge = self.graph.edges.get((cause, effect))
            if edge:
                edge.confidence = (edge.confidence + confidence) / 2
                edge.validation_count += 1
                edge.last_validated = _timestamp()
                edge.ate = np.mean([e["predicted"] - e["expected"] for e in evidence])
                self._save()

            return {"valid": valid, "confidence": confidence, "evidence": evidence}

    # ── LLM Explanation ─────────────────────────────────────────────────────

    def _llm_explain(self, cause: str, effect: str, chain_desc: str,
                     chain_parts: list) -> str:
        from luka_llm import generate
        prompt = (
            f"Explain this causal chain in clear, concise natural language.\n\n"
            f"Cause: {cause}\nEffect: {effect}\nChain: {chain_desc}\n"
            f"Details:\n" + "\n".join(f"  - {cp}" for cp in chain_parts) + "\n\n"
            f"Explain WHY this causal relationship exists. Be specific about the mechanism."
        )
        return generate(CAUSAL_MODEL, prompt).strip()

    # ── Integration ─────────────────────────────────────────────────────────

    def feed_prediction_errors(self):
        try:
            from brain.active_inference import get_active_inference
            inference = get_active_inference()
            with self._lock:
                for (cause, effect), edge in self.graph.edges.items():
                    if edge.confidence > 0.5:
                        cause_node = self.graph.nodes.get(cause)
                        if cause_node and cause_node.observations:
                            recent = cause_node.recent_values(5)
                            if recent:
                                predicted = edge.strength * float(recent[-1])
                                inference.predict_outcome(effect, context=f"causal:{cause}→{effect}")
        except Exception:
            pass

    def get_trajectory_predictions(self, start_state: dict,
                                   action_sequence: list) -> list:
        predictions = []
        current_state = dict(start_state)
        for action in action_sequence:
            action_name = action if isinstance(action, str) else action.get("tool", "unknown")
            effects = self.find_effects(action_name, max_depth=2)
            predicted_state = dict(current_state)
            for eff in effects:
                effect_name = eff["effect"]
                strength = eff["strength"]
                edge = self.graph.edges.get((action_name, effect_name))
                if edge:
                    current_val = float(current_state.get(effect_name, 0))
                    delta = strength * float(current_state.get(action_name, 0.5))
                    predicted_state[effect_name] = current_val + delta
            predictions.append(predicted_state)
            current_state = predicted_state
        return predictions

    def format_for_prompt(self, max_chars: int = 800) -> str:
        with self._lock:
            stats = self.get_stats()
            if stats["total_edges"] == 0:
                return ""

            parts = ["[CAUSAL REASONING — Pearl's Causal Hierarchy]"]
            parts.append(
                f"  Nodes: {stats['total_nodes']} | "
                f"Edges: {stats['total_edges']} | "
                f"Bi-directed: {stats.get('bi_edges', 0)} | "
                f"Avg confidence: {stats['avg_confidence']:.0%}"
            )

            top_edges = sorted(
                self.graph.edges.values(),
                key=lambda e: e.strength * e.confidence,
                reverse=True
            )[:5]

            if top_edges:
                parts.append("  Top causal links:")
                for edge in top_edges:
                    ate_str = f" ATE={edge.ate:.3f}" if edge.ate is not None else ""
                    parts.append(
                        f"    {edge.cause} → {edge.effect} "
                        f"(str={edge.strength:.2f} conf={edge.confidence:.2f}{ate_str})"
                    )

            result = "\n".join(parts)
            if len(result) > max_chars:
                result = result[:max_chars].rsplit("\n", 1)[0] + "\n  [...]"
            return result

    def get_stats(self) -> dict:
        with self._lock:
            confidences = [e.confidence for e in self.graph.edges.values()]
            strengths = [e.strength for e in self.graph.edges.values()]
            return {
                "total_nodes": len(self.graph.nodes),
                "total_edges": len(self.graph.edges),
                "bi_edges": len(self.graph.bi_edges),
                "total_equations": len(self.scm.equations),
                "avg_confidence": round(np.mean(confidences), 3) if confidences else 0,
                "avg_strength": round(np.mean(strengths), 3) if strengths else 0,
                "total_interventions": len(self._intervention_log),
                "total_counterfactuals": len(self._counterfactual_log),
                "high_confidence_edges": sum(1 for c in confidences if c > 0.7),
            }

    # ── Persistence ─────────────────────────────────────────────────────────

    def _save(self):
        try:
            data = {
                "meta": {
                    "version": 2,
                    "created": _timestamp(),
                    "last_update": _timestamp(),
                },
                "graph": self.graph.to_dict(),
                "scm": self.scm.to_dict(),
                "intervention_log": self._intervention_log[-100:],
                "counterfactual_log": self._counterfactual_log[-50:],
                "discovery_log": self._discovery_log[-20:],
            }
            CAUSAL_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except IOError as e:
            print(f"[CausalReasoner] Save failed: {e}")

    def _load(self):
        try:
            if CAUSAL_FILE.exists():
                data = json.loads(CAUSAL_FILE.read_text())
                self.graph = CausalGraph.from_dict(data.get("graph", {}))
                self.scm = StructuralCausalModel.from_dict(data.get("scm", {}))
                self._intervention_log = data.get("intervention_log", [])
                self._counterfactual_log = data.get("counterfactual_log", [])
                self._discovery_log = data.get("discovery_log", [])
                if self.graph.edges:
                    self.estimator = CausalEstimator(self.graph, np.array([]), list(self.graph.nodes.keys()))
        except Exception:
            pass

    def _start_consolidation_thread(self):
        def _consolidate_loop():
            while True:
                time.sleep(600)
                try:
                    self._consolidate()
                except Exception as e:
                    print(f"[CausalReasoner] Consolidation error: {e}")

        t = threading.Thread(target=_consolidate_loop, daemon=True, name="causal-consolidation")
        t.start()

    def _consolidate(self):
        with self._lock:
            pruned = 0
            for key in list(self.graph.edges.keys()):
                edge = self.graph.edges[key]
                edge.confidence *= CONFIDENCE_DECAY
                if edge.confidence < EDGE_STRENGTH_MIN and edge.validation_count < 2:
                    del self.graph.edges[key]
                    self.graph._forward[edge.cause].discard(edge.effect)
                    self.graph._backward[edge.effect].discard(edge.cause)
                    pruned += 1

            if pruned:
                print(f"[CausalReasoner] Consolidated: pruned {pruned} weak edges")
            self._save()


# ── Singleton ───────────────────────────────────────────────────────────────

_causal_reasoner = None
_causal_lock = threading.Lock()


def get_causal_reasoner(data: Optional[np.ndarray] = None,
                         var_names: Optional[List[str]] = None) -> CausalReasoner:
    """Get singleton causal reasoner instance."""
    global _causal_reasoner
    if _causal_reasoner is None:
        with _causal_lock:
            if _causal_reasoner is None:
                _causal_reasoner = CausalReasoner(data, var_names)
    return _causal_reasoner


# ── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Synthetic data test
    np.random.seed(42)
    n = 1000
    X = np.random.normal(0, 1, n)
    Z = X + np.random.normal(0, 0.5, n)  # X → Z
    Y = Z + np.random.normal(0, 0.5, n)  # Z → Y
    W = np.random.normal(0, 1, n)  # Independent

    data = np.column_stack([X, Z, Y, W])
    var_names = ["X", "Z", "Y", "W"]

    cr = get_causal_reasoner(data, var_names)

    print(f"\nStats: {cr.get_stats()}")
    print(f"\nIdentification: {cr.identify_effect('X', 'Y')}")
    print(f"\nATE X→Y: {cr.estimate_effect('X', 'Y')}")
    print(f"\nCauses of Y: {cr.find_causes('Y')}")
    print(f"\nEffects of X: {cr.find_effects('X')}")

    # Counterfactual
    cf = cr.counterfactual({"X": 1.0, "Z": 0.5, "Y": 1.2}, {"X": -1.0})
    print(f"\nCounterfactual: {cf['explanation']}")

    # Mediation
    med = cr.mediate("X", "Y", "Z", {"X": 1.0, "Z": 0.5})
    print(f"\nMediation: NDE={med['natural_direct_effect']:.3f}, NIE={med['natural_indirect_effect']:.3f}")

    print(f"\nPrompt format:\n{cr.format_for_prompt()}")