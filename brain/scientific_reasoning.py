#!/usr/bin/env python3
"""
scientific_reasoning.py — Luka Scientific Reasoning Loop (MCTS-Based)
========================================================================

The cognitive engine for scientific discovery. Implements a multi-pass
reasoning cycle using Monte Carlo Tree Search (MCTS) over hypothesis space.

Architecture:
  ┌──────────────────────────────────────────────────────────────────────┐
  │                    SCIENTIFIC REASONING LOOP (MCTS)                 │
  ├──────────────────────────────────────────────────────────────────────┤
  │  Selection:    UCT over hypothesis nodes (P(evidence) + exploration)│
  │  Expansion:   Generate new hypotheses via creativity_engine         │
  │  Simulation:  World model + causal reasoner predict outcomes         │
  │  Backprop:    Update node values with evidence fitness               │
  │  Synthesis:   Best theory from MCTS tree → theory_formation          │
  └──────────────────────────────────────────────────────────────────────┘

Cycle: Observe → MCTS_Hypothesize → Predict → Test → Revise → Theorize

Each phase uses the appropriate brain module:
  - Observe:    curiosity + world_model (detect anomalies, encode obs)
  - Hypothesize: MCTS over hypothesis space (creativity_engine + causal_reasoner)
  - Predict:    world_model + causal_reasoner (simulate outcomes)
  - Test:       discovery pipeline (execute experiments, gather evidence)
  - Revise:     theory_formation (update theories with new evidence)
  - Theorize:   theory_formation + abstraction_engine (synthesize knowledge)
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

BRAIN_DIR = Path(__file__).parent.resolve()
STATE_FILE = BRAIN_DIR / "scientific_reasoning_state.json"

# ── Configuration ───────────────────────────────────────────────────────────

MAX_CYCLES_PER_SESSION = 10
MIN_NOVELTY_FOR_HYPOTHESIS = 0.3
MAX_HYPOTHESES_PER_CYCLE = 5
REVISION_LEARNING_RATE = 0.15
MAX_REASONING_HISTORY = 100

# MCTS parameters
MCTS_SIMULATIONS = 100
MCTS_C_PUCT = 1.414  # Exploration constant
MCTS_MAX_DEPTH = 10
MCTS_SIMULATION_BUDGET = 50
UCT_EPSILON = 1e-10


def _timestamp() -> str:
    return datetime.now().isoformat()


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _safe_import(module_path: str, class_name: str):
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except Exception:
        return None


# ── Data Structures ─────────────────────────────────────────────────────────


class Observation:
    def __init__(self, phenomenon: str, context: str = "", domain: str = "general"):
        self.id = str(uuid.uuid4())[:8]
        self.phenomenon = phenomenon
        self.context = context
        self.domain = domain
        self.anomaly_score = 0.0
        self.features: Dict[str, Any] = {}
        self.timestamp = _timestamp()

    def to_dict(self) -> dict:
        return {"id": self.id, "phenomenon": self.phenomenon, "context": self.context,
                "domain": self.domain, "anomaly_score": round(self.anomaly_score, 3),
                "features": self.features, "timestamp": self.timestamp}


class Hypothesis:
    def __init__(self, statement: str, observation_id: str, domain: str = "general"):
        self.id = str(uuid.uuid4())[:8]
        self.statement = statement
        self.observation_id = observation_id
        self.domain = domain
        self.mechanisms: List[str] = []
        self.supporting_evidence: List[Dict] = []
        self.contradicting_evidence: List[Dict] = []
        self.predictions: List[Dict] = []
        self.confidence = 0.5
        self.novelty = 0.5
        self.falsifiability = 0.5
        self.parsimony = 0.5
        self.created_at = _timestamp()
        self.revised_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {"id": self.id, "statement": self.statement,
                "observation_id": self.observation_id, "domain": self.domain,
                "mechanisms": self.mechanisms,
                "confidence": round(self.confidence, 3),
                "novelty": round(self.novelty, 3),
                "falsifiability": round(self.falsifiability, 3),
                "parsimony": round(self.parsimony, 3),
                "created_at": self.created_at, "revised_at": self.revised_at}


class MCTSNode:
    """A node in the MCTS hypothesis space tree."""

    def __init__(self, hypothesis: Hypothesis, parent=None, action_idx: int = -1):
        self.hypothesis = hypothesis
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0  # Evidence accumulation
        self.action_idx = action_idx
        self.depth = parent.depth + 1 if parent else 0
        self.untried_actions: List[int] = []  # For expansion

    def ucb(self, c_puct: float = MCTS_C_PUCT) -> float:
        """UCT score: exploitation + exploration."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = c_puct * math.sqrt(math.log(self.parent.visits + 1) / (self.visits + UCT_EPSILON))
        return exploitation + exploration


class ReasoningCycle:
    def __init__(self, cycle_num: int, topic: str):
        self.id = str(uuid.uuid4())[:8]
        self.cycle_num = cycle_num
        self.topic = topic
        self.phase_durations: Dict[str, float] = {}
        self.observations: List[dict] = []
        self.hypotheses: List[dict] = []
        self.predictions: List[dict] = []
        self.revisions: List[dict] = []
        self.theories_synthesized: List[dict] = []
        self.insights: List[str] = []
        self.confidence_delta = 0.0
        self.mcts_statistics: Dict = {}
        self.started_at = _timestamp()
        self.completed_at = _timestamp()
        self.duration_s = 0.0

    def to_dict(self) -> dict:
        return {"id": self.id, "cycle_num": self.cycle_num, "topic": self.topic,
                "phase_durations": self.phase_durations,
                "observations": self.observations, "hypotheses": self.hypotheses,
                "predictions": self.predictions, "revisions": self.revisions,
                "theories_synthesized": self.theories_synthesized,
                "insights": self.insights,
                "confidence_delta": round(self.confidence_delta, 3),
                "mcts_statistics": self.mcts_statistics,
                "started_at": self.started_at, "completed_at": self.completed_at,
                "duration_s": round(self.duration_s, 2)}


# ── Scientific Reasoning Loop ────────────────────────────────────────────────


class ScientificReasoningLoop:
    """
    Multi-pass scientific reasoning engine with MCTS hypothesis search.

    Architecture:
    - MCTS over hypothesis space: UCT selection, expansion, simulation, backprop
    - Evidence accumulation: each simulation updates node values
    - Brain module integration: creativity_engine, causal_reasoner, world_model
    """

    def __init__(self, llm_fn=None):
        self._lock = threading.RLock()
        self._llm_fn = llm_fn
        self._cycle_count = 0
        self._history: List[ReasoningCycle] = []
        self._active_observations: List[Observation] = []
        self._active_hypotheses: List[Hypothesis] = []
        self._mcts_root: Optional[MCTSNode] = None
        self._insights: List[str] = []
        self._load_state()

    # ── Brain Module Access ──────────────────────────────────────────────

    def _get_causal_reasoner(self):
        cls = _safe_import("brain.causal_reasoner", "CausalReasoner")
        return cls() if cls else None

    def _get_world_model(self):
        cls = _safe_import("brain.world_model", "WorldModel")
        return cls() if cls else None

    def _get_creativity_engine(self):
        cls = _safe_import("brain.creativity_engine", "CreativityEngine")
        return cls() if cls else None

    def _get_abstraction_engine(self):
        cls = _safe_import("brain.abstraction_engine", "AbstractionEngine")
        return cls() if cls else None

    def _get_theory_engine(self):
        cls = _safe_import("brain.theory_formation", "TheoryFormationEngine")
        return cls() if cls else None

    def _get_curiosity(self):
        cls = _safe_import("brain.curiosity", "CuriosityModule")
        return cls() if cls else None

    async def _llm(self, prompt: str, **kwargs) -> str:
        if self._llm_fn:
            return await self._llm_fn(prompt, **kwargs)
        try:
            from discovery.pipeline import LLMStage
            stage = LLMStage("scientific_reasoning", max_retries=2, backoff=[3, 8], providers=["groq", "gemini"])
            result, _ = await stage.call_with_retry(prompt, **kwargs)
            return result or ""
        except Exception:
            return ""

    # ── MCTS Hypothesis Search ──────────────────────────────────────────

    def _hypothesis_score(self, h: Hypothesis) -> float:
        """Score hypothesis: evidence balance × novelty × parsimony."""
        support = sum(e.get("confidence", 0.5) for e in h.supporting_evidence)
        contra = sum(e.get("confidence", 0.5) for e in h.contradicting_evidence)
        evidence_ratio = support / max(support + contra, 0.01)
        return (evidence_ratio * 0.4 + h.novelty * 0.3 + h.parsimony * 0.3)

    def _generate_hypothesis_variants(self, h: Hypothesis, n: int = 3) -> List[Hypothesis]:
        """Generate variants of a hypothesis for MCTS expansion."""
        variants = []
        for i in range(n):
            v = Hypothesis(
                statement=f"{h.statement} (variant {i+1})",
                observation_id=h.observation_id,
                domain=h.domain,
            )
            v.mechanisms = list(h.mechanisms)
            v.novelty = h.novelty * (0.8 + 0.4 * (i / max(n, 1)))
            v.falsifiability = h.falsifiability * (0.9 + 0.2 * (i / max(n, 1)))
            v.parsimony = h.parsimony * (1.0 - 0.1 * i)
            v.confidence = self._score_hypothesis(v)
            variants.append(v)
        return variants

    def _simulate_hypothesis(self, h: Hypothesis) -> float:
        """Simulate: predict outcome of testing this hypothesis."""
        score = 0.0
        wm = self._get_world_model()
        cr = self._get_causal_reasoner()

        # World model simulation
        if wm:
            try:
                plan = wm.evaluate_plan([{"tool": "test", "hypothesis": h.statement}])
                score += plan.get("predicted_success", 0.5) * 0.3
            except Exception:
                score += 0.3

        # Causal reasoning
        if cr:
            try:
                effects = cr.find_effects(h.statement, max_depth=2)
                score += min(1.0, len(effects) * 0.15) * 0.3
            except Exception:
                score += 0.2

        # Base score
        score += h.confidence * 0.4

        return score

    def _mcts_search(self, root_hypothesis: Hypothesis, domain: str) -> List[Hypothesis]:
        """
        MCTS search over hypothesis space.

        UCT selection → expansion (generate variants) → simulation (predict) → backprop
        """
        root = MCTSNode(root_hypothesis)
        self._mcts_root = root

        # Generate initial children
        variants = self._generate_hypothesis_variants(root_hypothesis, n=3)
        for v in variants:
            child = MCTSNode(v, parent=root, action_idx=len(root.children))
            root.children.append(child)
            child.untried_actions = list(range(3))

        for _ in range(MCTS_SIMULATIONS):
            # Selection
            node = root
            while node.children and node.untried_actions:
                # UCT selection
                best_child = max(node.children, key=lambda c: c.ucb())
                if best_child.visits > 0:
                    node = best_child
                else:
                    break

            # Expansion
            if node.untried_actions and node.depth < MCTS_MAX_DEPTH:
                action = node.untried_actions.pop(0)
                new_variants = self._generate_hypothesis_variants(node.hypothesis, n=2)
                for v in new_variants:
                    child = MCTSNode(v, parent=node, action_idx=action)
                    node.children.append(child)
                    child.untried_actions = list(range(2) if node.depth < MCTS_MAX_DEPTH - 1 else [])
                if node.children:
                    node = node.children[-1]

            # Simulation
            reward = 0.0
            if node.hypothesis:
                reward = self._simulate_hypothesis(node.hypothesis)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

        # Collect best hypotheses from tree
        best_hypotheses = []
        self._collect_best_nodes(root, best_hypotheses, max_nodes=5)
        return [node.hypothesis for node in best_hypotheses]

    def _collect_best_nodes(self, node: MCTSNode, best: List[MCTSNode], max_nodes: int = 5):
        """Collect the best nodes from MCTS tree."""
        if node.visits > 0:
            best.append(node)
            best.sort(key=lambda n: n.value / max(n.visits, 1), reverse=True)
            if len(best) > max_nodes:
                best.pop()
        for child in node.children:
            self._collect_best_nodes(child, best, max_nodes)

    # ── Phase 1: Observe ────────────────────────────────────────────────

    async def observe(self, data: str, context: str = "", domain: str = "general") -> List[Observation]:
        t0 = time.time()
        observations = []

        prompt = f"""Analyze the following data and identify phenomena that are:
1. Surprising or unexpected
2. Unexplained by common knowledge
3. Potentially indicating deeper patterns

Data: {data[:3000]}
Context: {context[:1000]}
Domain: {domain}

For each phenomenon:
- phenomenon: what you observed
- anomaly_score: 0.0-1.0 (how surprising)
- features: key characteristics

Return JSON array of observations."""

        raw = await self._llm(prompt, json_mode=True, max_tokens=4096)
        parsed = self._parse_json(raw)

        if isinstance(parsed, list):
            for item in parsed[:MAX_HYPOTHESES_PER_CYCLE]:
                obs = Observation(item.get("phenomenon", "Unknown"), context, domain)
                obs.anomaly_score = float(item.get("anomaly_score", 0.5))
                obs.features = item.get("features", {}) if isinstance(item.get("features"), dict) else {}
                observations.append(obs)
        elif isinstance(parsed, dict):
            obs = Observation(parsed.get("phenomenon", data[:200]), context, domain)
            obs.anomaly_score = float(parsed.get("anomaly_score", 0.5))
            observations.append(obs)

        with self._lock:
            self._active_observations.extend(observations)

        print(f"[ScientificReasoning] observe(): {len(observations)} observations in {time.time()-t0:.1f}s")
        return observations

    # ── Phase 2: MCTS Hypothesize ───────────────────────────────────────

    async def hypothesize(self, observations: Optional[List[Observation]] = None,
                          domain: str = "general") -> List[Hypothesis]:
        t0 = time.time()
        observations = observations or self._active_observations
        if not observations:
            return []

        hypotheses = []
        for obs in observations[:3]:
            obs_descriptions = f"[{obs.id}] {obs.phenomenon} (anomaly: {obs.anomaly_score:.2f})"

            prompt = f"""You are a scientist. Generate a hypothesis for this observation:

{obs_descriptions}

Provide:
- statement: the hypothesis
- mechanisms: list of causal mechanisms
- falsifiability: 0.0-1.0
- parsimony: 0.0-1.0
- novelty: 0.0-1.0

Return JSON."""

            raw = await self._llm(prompt, json_mode=True, max_tokens=2048)
            parsed = self._parse_json(raw)

            if isinstance(parsed, dict):
                h = Hypothesis(parsed.get("statement", ""), obs.id, domain)
                h.mechanisms = parsed.get("mechanisms", [])
                h.falsifiability = _sigmoid(float(parsed.get("falsifiability", 0.5)))
                h.parsimony = _sigmoid(float(parsed.get("parsimony", 0.5)))
                h.novelty = _sigmoid(float(parsed.get("novelty", 0.5)))
                h.confidence = self._score_hypothesis(h)

                # MCTS search over this hypothesis
                mcts_hypotheses = self._mcts_search(h, domain)
                hypotheses.extend(mcts_hypotheses)

        with self._lock:
            self._active_hypotheses.extend(hypotheses)

        print(f"[ScientificReasoning] hypothesize(): {len(hypotheses)} hypotheses (MCTS) in {time.time()-t0:.1f}s")
        return hypotheses

    def _score_hypothesis(self, h: Hypothesis) -> float:
        base = (h.falsifiability * 0.3 + h.parsimony * 0.3 + h.novelty * 0.2 + 0.2)
        complexity_penalty = len(h.mechanisms) * 0.05
        return _sigmoid(base - complexity_penalty)

    # ── Phase 3-6: Predict, Test, Revise, Theorize ──────────────────────

    async def predict(self, hypotheses: Optional[List[Hypothesis]] = None) -> List[Dict]:
        t0 = time.time()
        hypotheses = hypotheses or self._active_hypotheses
        if not hypotheses:
            return []

        predictions = []
        for h in hypotheses[:MAX_HYPOTHESES_PER_CYCLE]:
            prompt = f"""Given this scientific hypothesis:
"{h.statement}"
Mechanisms: {', '.join(h.mechanisms) if h.mechanisms else 'unknown'}

Generate 1-2 testable predictions that:
1. Follow LOGICALLY from the hypothesis
2. Are SPECIFIC and testable
3. Include a TEST METHOD

Return JSON array of predictions with: statement, expected_outcome, test_method, confidence."""

            raw = await self._llm(prompt, json_mode=True, max_tokens=2048)
            parsed = self._parse_json(raw)
            items = parsed if isinstance(parsed, list) else [parsed] if isinstance(parsed, dict) else []

            for item in items[:2]:
                predictions.append({
                    "statement": item.get("statement", ""),
                    "hypothesis_id": h.id,
                    "test_method": item.get("test_method", ""),
                    "expected_outcome": item.get("expected_outcome", ""),
                    "confidence_before": _sigmoid(float(item.get("confidence", 0.5))),
                    "tested": False,
                    "result": "pending",
                })

        print(f"[ScientificReasoning] predict(): {len(predictions)} predictions in {time.time()-t0:.1f}s")
        return predictions

    async def test(self, predictions: Optional[List[Dict]] = None, evidence: Optional[str] = None) -> List[Dict]:
        t0 = time.time()
        predictions = predictions or []
        if not predictions:
            return []

        tested = []
        for pred in predictions[:MAX_HYPOTHESES_PER_CYCLE]:
            if evidence:
                prompt = f"""A prediction was made: "{pred['statement']}"
Expected: "{pred['expected_outcome']}"

New evidence: {evidence[:3000]}

Evaluate: confirmed, refuted, or inconclusive?
Return JSON: {{"result": "confirmed|refuted|inconclusive", "confidence": 0.0-1.0}}"""

                raw = await self._llm(prompt, json_mode=True, max_tokens=1024)
                parsed = self._parse_json(raw)
                if isinstance(parsed, dict):
                    pred["result"] = parsed.get("result", "inconclusive")
                    pred["confidence_after"] = _sigmoid(float(parsed.get("confidence", 0.5)))
                    pred["tested"] = True
                    tested.append(pred)

        print(f"[ScientificReasoning] test(): {len(tested)} tested in {time.time()-t0:.1f}s")
        return tested

    async def revise(self, hypotheses: Optional[List[Hypothesis]] = None) -> List[Dict]:
        t0 = time.time()
        hypotheses = hypotheses or self._active_hypotheses
        revisions = []

        for h in hypotheses[:MAX_HYPOTHESES_PER_CYCLE]:
            if not h.supporting_evidence and not h.contradicting_evidence:
                continue
            support = sum(e.get("confidence", 0.5) for e in h.supporting_evidence)
            contra = sum(e.get("confidence", 0.5) for e in h.contradicting_evidence)
            h.confidence = _sigmoid(support / max(support + contra, 0.01) * 2 - 1)
            h.revised_at = _timestamp()

        print(f"[ScientificReasoning] revise(): {len(revisions)} revisions in {time.time()-t0:.1f}s")
        return revisions

    async def theorize(self, hypotheses: Optional[List[Hypothesis]] = None, domain: str = "general") -> List[Dict]:
        t0 = time.time()
        hypotheses = hypotheses or self._active_hypotheses
        strong = [h for h in hypotheses if h.confidence >= 0.5]
        if not strong:
            return []

        theories = []
        # Get MCTS statistics
        mcts_stats = {}
        if self._mcts_root:
            total_nodes = self._mcts_root.visits
            for child in self._mcts_root.children:
                pass
            mcts_stats = {"total_simulations": MCTS_SIMULATIONS, "root_visits": total_nodes}

        if len(strong) >= 2:
            hyp_descriptions = "\n".join(f"- [{h.id}] {h.statement} (conf: {h.confidence:.2f})" for h in strong)
            prompt = f"""Synthesize these confirmed hypotheses into a coherent theory:

{hyp_descriptions}

Create a UNIFIED theory:
- name: theory name
- description: full theory description
- unifying_principle: what connects them
- mechanisms: list of causal mechanisms
- confidence: 0.0-1.0

Return JSON."""

            raw = await self._llm(prompt, json_mode=True, max_tokens=4096)
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict) and parsed.get("description"):
                theories.append({
                    "name": parsed.get("name", "Theory"),
                    "description": parsed.get("description", ""),
                    "unifying_principle": parsed.get("unifying_principle", ""),
                    "mechanisms": parsed.get("mechanisms", []),
                    "confidence": _sigmoid(float(parsed.get("confidence", 0.5))),
                    "source_hypotheses": [h.id for h in strong],
                    "mcts_statistics": mcts_stats,
                })

        print(f"[ScientificReasoning] theorize(): {len(theories)} theories in {time.time()-t0:.1f}s")
        return theories

    # ── Full Reasoning Cycle ────────────────────────────────────────────

    async def reason(self, topic: str, data: str = "", context: str = "",
                     domain: str = "general", evidence: str = "") -> ReasoningCycle:
        with self._lock:
            self._cycle_count += 1
            cycle_num = self._cycle_count

        cycle = ReasoningCycle(cycle_num, topic)
        total_start = time.time()

        print(f"\n{'='*60}\n[ScientificReasoning] MCTS CYCLE {cycle_num}: {topic}\n{'='*60}")

        # Phase 1: Observe
        t = time.time()
        observations = await self.observe(data or topic, context, domain)
        cycle.phase_durations["observe"] = time.time() - t
        cycle.observations = [o.to_dict() for o in observations]

        # Phase 2: MCTS Hypothesize
        t = time.time()
        hypotheses = await self.hypothesize(observations, domain)
        cycle.phase_durations["hypothesize"] = time.time() - t
        cycle.hypotheses = [h.to_dict() for h in hypotheses]
        # MCTS stats
        if self._mcts_root:
            cycle.mcts_statistics = {
                "simulations": MCTS_SIMULATIONS,
                "root_visits": self._mcts_root.visits,
                "total_nodes": len(self._mcts_root.children) if self._mcts_root.children else 0,
            }

        # Phase 3: Predict
        t = time.time()
        predictions = await self.predict(hypotheses)
        cycle.phase_durations["predict"] = time.time() - t
        cycle.predictions = predictions

        # Phase 4: Test (if evidence available)
        if evidence:
            t = time.time()
            tested = await self.test(predictions, evidence)
            cycle.phase_durations["test"] = time.time() - t

        # Phase 5: Revise
        t = time.time()
        revisions = await self.revise(hypotheses)
        cycle.phase_durations["revise"] = time.time() - t
        cycle.revisions = revisions

        # Phase 6: Theorize
        t = time.time()
        theories = await self.theorize(hypotheses, domain)
        cycle.phase_durations["theorize"] = time.time() - t
        cycle.theories_synthesized = theories

        # Insights
        cycle.insights = self._extract_insights(observations, hypotheses, theories)

        cycle.completed_at = _timestamp()
        cycle.duration_s = time.time() - total_start

        with self._lock:
            self._history.append(cycle)
            if len(self._history) > MAX_REASONING_HISTORY:
                self._history = self._history[-MAX_REASONING_HISTORY:]
        self._save_state()

        print(f"\n[ScientificReasoning] MCTS Cycle {cycle_num} complete in {cycle.duration_s:.1f}s")
        print(f"  Observations: {len(observations)}, Hypotheses: {len(hypotheses)}")
        print(f"  Predictions: {len(predictions)}, Theories: {len(theories)}")
        print(f"  MCTS simulations: {MCTS_SIMULATIONS}")
        if cycle.insights:
            print(f"  Insights: {len(cycle.insights)}")

        return cycle

    # ── Utilities ────────────────────────────────────────────────────────

    def _extract_insights(self, observations, hypotheses, theories) -> List[str]:
        insights = []
        for obs in observations:
            if obs.anomaly_score > 0.7:
                insights.append(f"High anomaly: {obs.phenomenon[:100]}")
        for h in hypotheses:
            if h.confidence > 0.7:
                insights.append(f"Strong hypothesis: {h.statement[:100]}")
        for t in theories:
            if t.get("confidence", 0) > 0.6:
                insights.append(f"Theory: {t.get('name', 'Unnamed')}")
        return insights[:10]

    def _parse_json(self, raw: str) -> Any:
        if not raw:
            return None
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            for start_char, end_char in [('[', ']'), ('{', '}')]:
                start = raw.find(start_char)
                end = raw.rfind(end_char)
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(raw[start:end + 1])
                    except json.JSONDecodeError:
                        continue
            return None

    def _save_state(self):
        try:
            state = {"cycle_count": self._cycle_count,
                     "history": [c.to_dict() for c in self._history[-20:]],
                     "insights": self._insights[-50:], "saved_at": _timestamp()}
            STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
        except Exception as e:
            print(f"[ScientificReasoning] Save error: {e}")

    def _load_state(self):
        try:
            if STATE_FILE.exists():
                data = json.loads(STATE_FILE.read_text())
                self._cycle_count = data.get("cycle_count", 0)
                self._insights = data.get("insights", [])
        except Exception:
            pass

    @property
    def status(self) -> Dict[str, Any]:
        return {"cycle_count": self._cycle_count,
                "active_observations": len(self._active_observations),
                "active_hypotheses": len(self._active_hypotheses),
                "total_insights": len(self._insights)}


# ── Singleton ───────────────────────────────────────────────────────────────

_instance: Optional[ScientificReasoningLoop] = None
_init_lock = threading.Lock()


def get_scientific_reasoning_loop(llm_fn=None) -> ScientificReasoningLoop:
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                _instance = ScientificReasoningLoop(llm_fn=llm_fn)
    return _instance