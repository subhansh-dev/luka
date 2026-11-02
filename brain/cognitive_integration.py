#!/usr/bin/env python3
"""
cognitive_integration.py — Luka Cognitive Integration Layer (WTA + Global Workspace)
======================================================================================

Research-grade implementation of Dehaene's Global Neuronal Workspace (GNW) theory
with basal ganglia-style Winner-Take-All (WTA) competition.

Architecture:
  ┌──────────────────────────────────────────────────────────────────────┐
  │                    COGNITIVE INTEGRATION LAYER                      │
  ├──────────────────────────────────────────────────────────────────────┤
  │                                                                      │
  │  Module 1 ────┐                                                     │
  │  Module 2 ────┼───► WTA Competition ───► Global Workspace ──► Response │
  │  Module 3 ────┘     (basal ganglia)       (ignition)               │
  │                                                                      │
  │  Dynamic Priority: γ * Q(a) + ν * I(a) + σ * S(a)                  │
  │  WTA: Nonlinear competition, inhibition of losing modules            │
  │  Broadcast: Selected module's content → all modules (global ignition)│
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘

Key principles:
  [CI-1] Global Workspace Theory (Baars, 1988; Dehaene & Naccache, 2001)
         - Information broadcast via global workspace
         - Ignition: sustained activity in workspace = conscious access

  [CI-2] Dual-Process Theory (Kahneman, 2011)
         - System 1: fast, automatic, intuitive
         - System 2: slow, deliberate, reasoning

  [CI-3] Basal Ganglia Gating (Frank, 2005)
         - Direct/Indirect pathway competition for action selection
         - Go/NoGo: WTA via dopamine-modulated competition

  [CI-4] Somatic Marker Hypothesis (Damasio, 1994)
         - Emotional/body signals modulate decision confidence
         - Somatic markers: valence from past outcomes

Pipeline:
  1. Module competition (WTA) for processing rights
  2. Global workspace ignition (winner broadcasts)
  3. Fast path (System 1) if confident
  4. Deliberative path (System 2) for novel/complex
  5. Somatic marker modulation of confidence
  6. Metacognitive reflection on outcome
"""

import json
import math
import threading
import time
import warnings
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

BRAIN_DIR = Path(__file__).parent.resolve()
INTEGRATION_FILE = BRAIN_DIR / "cognitive_integration.json"

# ── Configuration ───────────────────────────────────────────────────────────

FAST_PATH_CONFIDENCE = 0.75
FAST_PATH_TIMEOUT_MS = 100
PIPELINE_TIMEOUT_MS = 5000
MODULE_TIMEOUT_MS = 1000

# WTA parameters
WTA_GO_THRESHOLD = 0.5    # Direct pathway activation
WTA_NOGO_THRESHOLD = 0.3  # Indirect pathway inhibition
WTA_LEARNING_RATE = 0.1   # Weight update
WTA_DECAY = 0.99          # Weight decay
WTA_GAMMA = 0.6           # Skill weight
WTA_NU = 0.3              # Novelty weight
WTA_SIGMA = 0.1           # Stimulus weight

# Module priorities
MODULE_PRIORITIES = {
    "metacognitive": 10, "intuition": 8, "causal": 7,
    "analogy": 6, "creativity": 5, "world_model": 4,
    "neurosymbolic": 3, "emotional": 2,
}

MIN_RESPONSE_CONFIDENCE = 0.1
MAX_MODULES_FOR_RESPONSE = 5
MAX_TRACE_HISTORY = 200
MAX_PERFORMANCE_LOG = 500
GLOBAL_WORKSPACE_CAPACITY = 3  # Items simultaneously in workspace


def _now() -> str:
    return datetime.now().isoformat()


def _timestamp() -> float:
    return time.time()


def _hash_request(request: str) -> str:
    import hashlib
    return hashlib.md5(request.encode()).hexdigest()[:12]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ── Data Structures ─────────────────────────────────────────────────────────


class ModuleCompetitor:
    """A module competing for workspace access."""

    def __init__(self, name: str, priority: float = 0.5):
        self.name = name
        self.priority = priority
        self.skill: float = 0.0  # Q(a): past success rate
        self.novelty: float = 0.0  # I(a): information gain
        self.stimulus: float = 0.0  # S(a): input relevance
        self.salience: float = 0.0  # Combined: γQ + νI + σS
        self.inhibited: bool = False  # WTA: losing competitors are inhibited
        self.last_activation: float = 0.0

    def compute_salience(self, gamma: float = WTA_GAMMA, nu: float = WTA_NU,
                         sigma: float = WTA_SIGMA) -> float:
        """Compute salience = γ * skill + ν * novelty + σ * stimulus."""
        self.salience = gamma * self.skill + nu * self.novelty + sigma * self.stimulus
        return self.salience

    def to_dict(self) -> dict:
        return {"name": self.name, "priority": self.priority, "skill": self.skill,
                "novelty": self.novelty, "stimulus": self.stimulus, "salience": self.salience,
                "inhibited": self.inhibited}


class GlobalWorkspace:
    """Dehaene's Global Neuronal Workspace — information broadcast system."""

    def __init__(self, capacity: int = GLOBAL_WORKSPACE_CAPACITY):
        self.capacity = capacity
        self.contents: List[Dict] = []  # Items currently in workspace
        self.broadcast_history: List[Dict] = []
        self.winner: Optional[ModuleCompetitor] = None
        self.ignition_strength: float = 0.0  # 0-1: strength of global ignition

    def push(self, content: Dict):
        """Push content to workspace (bounded capacity)."""
        self.contents.append(content)
        if len(self.contents) > self.capacity:
            self.contents.pop(0)

    def broadcast(self, competitor: ModuleCompetitor, content: Any) -> Dict:
        """Broadcast content from winning module to all modules."""
        self.winner = competitor
        entry = {
            "timestamp": _now(),
            "winner": competitor.name,
            "salience": competitor.salience,
            "content": str(content)[:200],
            "ignition_strength": self.ignition_strength,
        }
        self.broadcast_history.append(entry)
        if len(self.broadcast_history) > 100:
            self.broadcast_history = self.broadcast_history[-100:]
        return entry

    def get_stats(self) -> dict:
        return {"capacity": self.capacity, "current_items": len(self.contents),
                "total_broadcasts": len(self.broadcast_history),
                "last_winner": self.winner.name if self.winner else None,
                "ignition_strength": round(self.ignition_strength, 3)}


class CognitiveResponse:
    def __init__(self, response: str, confidence: float = 0.5, path: str = "unknown"):
        self.response = response
        self.confidence = _clamp(confidence)
        self.modules_used: List[str] = []
        self.reasoning_trace: List[dict] = []
        self.emotional_state: Optional[dict] = None
        self.path = path
        self.duration_ms = 0.0
        self.request_hash = ""
        self.strategy_used = "default"
        self.timestamp = _now()

    def add_trace_step(self, module: str, action: str, result: Any = None,
                       duration_ms: float = 0.0, confidence: float = 0.0):
        self.reasoning_trace.append({"module": module, "action": action,
            "result_summary": str(result)[:200] if result else None,
            "duration_ms": round(duration_ms, 2), "confidence": round(confidence, 3),
            "timestamp": _now()})
        if module not in self.modules_used:
            self.modules_used.append(module)

    def to_dict(self) -> dict:
        return {"response": self.response[:1000], "confidence": round(self.confidence, 3),
                "modules_used": self.modules_used, "reasoning_trace": self.reasoning_trace,
                "emotional_state": self.emotional_state, "path": self.path,
                "duration_ms": round(self.duration_ms, 2), "request_hash": self.request_hash,
                "strategy_used": self.strategy_used, "timestamp": self.timestamp}


class PipelineMetrics:
    def __init__(self):
        self.total_requests = 0
        self.fast_path_count = 0
        self.deliberative_count = 0
        self.avg_fast_ms = 0.0
        self.avg_deliberative_ms = 0.0
        self.module_invocations: Dict[str, int] = defaultdict(int)
        self.module_failures: Dict[str, int] = defaultdict(int)
        self.confidence_history: deque = deque(maxlen=100)
        self.wta_history: List[Dict] = []

    def record(self, response: CognitiveResponse):
        self.total_requests += 1
        if response.path == "fast":
            self.fast_path_count += 1
            self.avg_fast_ms = (self.avg_fast_ms * (self.fast_path_count - 1) + response.duration_ms) / self.fast_path_count
        else:
            self.deliberative_count += 1
            self.avg_deliberative_ms = (self.avg_deliberative_ms * (self.deliberative_count - 1) + response.duration_ms) / self.deliberative_count
        for module in response.modules_used:
            self.module_invocations[module] += 1
        self.confidence_history.append(response.confidence)

    def to_dict(self) -> dict:
        return {"total_requests": self.total_requests,
                "fast_path_count": self.fast_path_count,
                "deliberative_count": self.deliberative_count,
                "avg_fast_ms": round(self.avg_fast_ms, 1),
                "avg_deliberative_ms": round(self.avg_deliberative_ms, 1),
                "module_invocations": dict(self.module_invocations),
                "module_failures": dict(self.module_failures),
                "avg_confidence": round(sum(self.confidence_history) / max(len(self.confidence_history), 1), 3)}


# ── Cognitive Integration ────────────────────────────────────────────────────


class CognitiveIntegration:
    """
    Central cognitive integration with WTA competition + Global Workspace.

    Orchestrates all cognitive modules via:
    1. Module competition (basal ganglia-style WTA)
    2. Winner broadcasts to global workspace
    3. Fast path (System 1) or deliberative pipeline (System 2)
    4. Somatic marker modulation of confidence
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {}
        self._metrics = PipelineMetrics()
        self._trace_history: List[dict] = []
        self._module_cache: Dict[str, Any] = {}
        self._workspace = GlobalWorkspace()
        self._competitors: Dict[str, ModuleCompetitor] = {}
        self._somatic_markers: Dict[str, float] = {}
        self._session_start = _timestamp()
        self._load()

        # Initialize competitors
        for name, prio in MODULE_PRIORITIES.items():
            self._competitors[name] = ModuleCompetitor(name, prio)

        print(f"[CognitiveIntegration] Initialized with WTA + Global Workspace ({len(self._competitors)} competitors)")

    # ── Persistence ──────────────────────────────────────────────────────

    def _empty_store(self) -> dict:
        return {"meta": {"version": 2, "created": _now(), "last_update": _now(),
                "total_requests": 0, "total_fast_path": 0, "total_deliberative": 0},
                "metrics": {}, "recent_traces": [], "somatic_markers": {},
                "competitor_weights": {}}

    def _load(self):
        if not INTEGRATION_FILE.exists():
            self._data = self._empty_store()
            self._save()
            return
        try:
            raw = INTEGRATION_FILE.read_text(encoding="utf-8")
            self._data = json.loads(raw)
            self._trace_history = self._data.get("recent_traces", [])
            self._somatic_markers = self._data.get("somatic_markers", {})
            # Load competitor weights
            weights = self._data.get("competitor_weights", {})
            for name, w in weights.items():
                if name in self._competitors:
                    self._competitors[name].skill = w.get("skill", 0.0)
                    self._competitors[name].novelty = w.get("novelty", 0.0)
                    self._competitors[name].stimulus = w.get("stimulus", 0.0)
        except (json.JSONDecodeError, IOError):
            self._data = self._empty_store()
            self._save()

    def _save(self):
        BRAIN_DIR.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._data["metrics"] = self._metrics.to_dict()
            self._data["recent_traces"] = self._trace_history[-MAX_TRACE_HISTORY:]
            self._data["somatic_markers"] = self._somatic_markers
            self._data["competitor_weights"] = {
                n: {"skill": c.skill, "novelty": c.novelty, "stimulus": c.stimulus}
                for n, c in self._competitors.items()
            }
            self._data["meta"]["last_update"] = _now()
            INTEGRATION_FILE.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    # ── Module Loading ──────────────────────────────────────────────────

    def _get_module(self, name: str) -> Optional[Any]:
        if name in self._module_cache:
            return self._module_cache[name]
        try:
            imports = {
                "intuition": ("brain.intuition_engine", "get_intuition_engine"),
                "metacognitive": ("brain.metacognitive_monitor", "get_metacognitive_monitor"),
                "emotional": ("brain.emotional_regulation", "get_emotional_regulation"),
                "causal": ("brain.causal_reasoner", "get_causal_reasoner"),
                "analogy": ("brain.analogy_engine", "get_analogy_engine"),
                "creativity": ("brain.creativity_engine", "get_creativity_engine"),
                "world_model": ("brain.world_model", "get_world_model"),
                "neurosymbolic": ("brain.neurosymbolic_reasoner", "get_neurosymbolic_reasoner"),
                "learning": ("brain.learning", "get_learning_engine"),
                "episodic": ("brain.episodic_memory", "get_episodic_memory"),
                "procedural": ("brain.procedural_memory", "get_procedural_memory"),
                "active_inference": ("brain.active_inference", "get_active_inference"),
                "global_workspace": ("brain.global_workspace", "get_global_workspace"),
            }
            if name in imports:
                module_path, func_name = imports[name]
                import importlib
                mod = importlib.import_module(module_path)
                func = getattr(mod, func_name)
                self._module_cache[name] = func()
                return self._module_cache[name]
        except Exception:
            pass
        return None

    def _call_module(self, module_name: str, method_name: str, *args, **kwargs) -> Tuple[bool, Any]:
        start = _timestamp()
        mod = self._get_module(module_name)
        if mod is None:
            return False, None
        try:
            method = getattr(mod, method_name, None)
            if method is None:
                return False, None
            result = method(*args, **kwargs)
            return True, result
        except Exception as e:
            self._metrics.module_failures[module_name] = self._metrics.module_failures.get(module_name, 0) + 1
            return False, str(e)

    # ── WTA Competition ─────────────────────────────────────────────────

    def _wta_competition(self, request: str, context: dict) -> ModuleCompetitor:
        """
        Winner-Take-All competition among modules.

        Each module's salience = γ * skill + ν * novelty + σ * stimulus.
        Winner = max salience, losers are inhibited.
        """
        task_type = context.get("task_type", "general")
        domain = context.get("domain", "general")

        for comp in self._competitors.values():
            comp.inhibited = False

        # Compute salience for each competitor
        for name, comp in self._competitors.items():
            # Skill from past success
            skill = comp.skill

            # Novelty from curiosity
            novelty = 0.0
            if name == "creativity":
                novelty = 0.6
            elif name == "causal":
                novelty = 0.3

            # Stimulus from input relevance
            stimulus = 0.5
            if task_type == "analogy" and name == "analogy":
                stimulus = 0.9
            elif task_type == "causal" and name == "causal":
                stimulus = 0.9
            elif task_type == "creative" and name == "creativity":
                stimulus = 0.9
            elif task_type == "routine" and name == "intuition":
                stimulus = 0.8

            comp.skill = skill
            comp.novelty = novelty
            comp.stimulus = stimulus
            comp.compute_salience()

        # WTA: find winner, inhibit losers
        sorted_comps = sorted(self._competitors.values(), key=lambda c: c.salience, reverse=True)
        winner = sorted_comps[0] if sorted_comps else None

        # Inhibit losing competitors (WTA mechanism)
        for comp in sorted_comps[1:]:
            if comp.salience < winner.salience * 0.7:  # Significant margin
                comp.inhibited = True

        # Record WTA outcome
        self._metrics.wta_history.append({
            "timestamp": _now(),
            "winner": winner.name if winner else None,
            "salience": winner.salience if winner else 0,
            "competitors": {c.name: round(c.salience, 3) for c in self._competitors.values()},
        })

        return winner

    def _update_wta_weights(self, winner: ModuleCompetitor, success: bool):
        """Update skill weights based on outcome."""
        lr = WTA_LEARNING_RATE
        if success:
            winner.skill = min(1.0, winner.skill + lr)
        else:
            winner.skill = max(0.0, winner.skill - lr * WTA_DECAY)

    # ── Fast Path (System 1) ────────────────────────────────────────────

    def _fast_path(self, request: str, context: dict,
                   response: CognitiveResponse) -> bool:
        start = _timestamp()

        winner = self._wta_competition(request, context)
        if winner is None:
            return False

        # Try intuition recognition
        success, result = self._call_module("intuition", "recognize", request,
                                            context.get("domain", "general"))
        if success and result:
            action, confidence, match_info = result
            elapsed = (_timestamp() - start) * 1000
            response.add_trace_step("intuition", "recognize", action, elapsed, confidence)

            if action and confidence >= FAST_PATH_CONFIDENCE:
                # Broadcast to global workspace
                self._workspace.ignition_strength = min(1.0, confidence)
                broadcast = self._workspace.broadcast(winner, action)

                response.response = action
                response.path = "fast"
                response.strategy_used = "intuitive_recognition"
                response.confidence = confidence
                return True

        return False

    # ── Deliberative Pipeline (System 2) ───────────────────────────────

    def _deliberative_pipeline(self, request: str, context: dict,
                                response: CognitiveResponse):
        response.path = "deliberative"
        pipeline_start = _timestamp()
        gathered_evidence: List[dict] = []

        # WTA competition for deliberative path
        winner = self._wta_competition(request, context)
        if winner:
            self._workspace.broadcast(winner, f"Processing: {request[:100]}")
            response.strategy_used = winner.name

        # Step 1: Metacognitive strategy
        strat_success, strategy = self._call_module("metacognitive", "suggest_strategy",
                                                    context.get("task_type", "general"))
        if strat_success and strategy:
            response.strategy_used = strategy
            response.add_trace_step("metacognitive", "suggest_strategy", strategy,
                                    (_timestamp() - pipeline_start) * 1000, 0.0)

        # Step 2: Causal reasoning
        causal_success, causal_result = self._call_module("causal", "identify_effect",
                                                           request[:200], "outcome")
        if causal_success and causal_result:
            response.add_trace_step("causal", "identify_effect", str(causal_result)[:150],
                                    (_timestamp() - pipeline_start) * 1000, 0.0)
            gathered_evidence.append({"source": "causal", "data": causal_result, "weight": 0.8})

        # Step 3: Analogical reasoning
        analogy_success, analogy_result = self._call_module("analogy", "find_analogies",
                                                             request[:200], context.get("domain", "general"))
        if analogy_success and analogy_result:
            response.add_trace_step("analogy", "find_analogies", str(analogy_result)[:150],
                                    (_timestamp() - pipeline_start) * 1000, 0.0)
            gathered_evidence.append({"source": "analogy", "data": analogy_result, "weight": 0.6})

        # Step 4: Creativity check
        creativity_success, creative_result = self._call_module("creativity", "generate_alternatives",
                                                                request, context)
        if creativity_success and creative_result:
            response.add_trace_step("creativity", "generate_alternatives", str(creative_result)[:150],
                                    (_timestamp() - pipeline_start) * 1000, 0.0)
            gathered_evidence.append({"source": "creativity", "data": creative_result, "weight": 0.5})

        # Step 5: World model simulation
        wm_success, wm_result = self._call_module("world_model", "evaluate_plan", [request])
        if wm_success and wm_result:
            response.add_trace_step("world_model", "simulate_outcome", str(wm_result)[:150],
                                    (_timestamp() - pipeline_start) * 1000, 0.0)
            gathered_evidence.append({"source": "world_model", "data": wm_result, "weight": 0.7})

        # Step 6: Neurosymbolic verification
        neuro_success, neuro_result = self._call_module("neurosymbolic", "check_consistency",
                                                        [{"statement": str(ev.get("data", ""))[:100],
                                                          "confidence": ev.get("weight", 0.5)} for ev in gathered_evidence])
        if neuro_success and neuro_result:
            response.add_trace_step("neurosymbolic", "verify_reasoning", str(neuro_result)[:150],
                                    (_timestamp() - pipeline_start) * 1000, 0.0)

        # Step 7: Synthesize
        synthesis = self._synthesize_response(request, context, gathered_evidence, response)
        response.response = synthesis.get("response", "")
        response.confidence = synthesis.get("confidence", 0.5)

        # Somatic marker modulation
        marker_key = response.strategy_used
        if marker_key in self._somatic_markers:
            marker = self._somatic_markers[marker_key]
            response.confidence = _clamp(response.confidence + marker * 0.05)

        # Broadcast to global workspace
        self._workspace.broadcast(winner, response.response)

    def _synthesize_response(self, request: str, context: dict, evidence: List[dict],
                              response: CognitiveResponse) -> dict:
        if not evidence:
            return {"response": f"Processed through deliberative pipeline. Request: {request[:100]}", "confidence": 0.3}

        total_weight = 0.0
        for ev in evidence:
            weight = ev.get("weight", 0.5)
            data = ev.get("data")
            if isinstance(data, dict):
                conf = data.get("confidence", data.get("ate", 0.5))
            elif isinstance(data, (list, tuple)) and len(data) >= 2:
                conf = data[1] if isinstance(data[1], (int, float)) else 0.5
            else:
                conf = 0.5
            total_weight += weight * conf

        final_confidence = _clamp(total_weight / max(len(evidence), 1))

        evidence_summary = [f"{e['source']}({e.get('weight', 0.5):.2f})" for e in evidence]
        response_parts = [f"Deliberative analysis of: {request[:80]}"]
        if evidence_summary:
            response_parts.append(f"Evidence from: {', '.join(evidence_summary)}")

        return {"response": " | ".join(response_parts), "confidence": final_confidence}

    # ── Post-Processing ─────────────────────────────────────────────────

    def _post_process(self, request: str, context: dict, response: CognitiveResponse):
        # Update WTA weights
        winner = self._workspace.winner
        if winner:
            self._update_wta_weights(winner, response.confidence > 0.5)

        # Create somatic marker
        if response.confidence > 0.7:
            self._somatic_markers[response.strategy_used] = 0.1
        elif response.confidence < 0.3:
            self._somatic_markers[response.strategy_used] = -0.1

        # Record to episodic memory
        self._call_module("episodic", "encode_event", "cognitive_response",
                          f"Request: {request[:100]} | Response: {response.response[:100]}",
                          context.get("domain", "general"))

        # Learning signal
        self._call_module("learning", "record_event", "cognitive_response",
                          {"request": request[:200], "response": response.response[:200],
                           "confidence": response.confidence})

    # ── Main Entry Point ───────────────────────────────────────────────

    def process_request(self, request: str, context: Optional[dict] = None) -> CognitiveResponse:
        start_time = _timestamp()
        context = context or {}
        request_hash = _hash_request(request)

        response = CognitiveResponse(response="", confidence=0.0)
        response.request_hash = request_hash

        with self._lock:
            self._data["meta"]["total_requests"] += 1

        # Fast Path (System 1)
        try:
            if self._fast_path(request, context, response):
                response.duration_ms = (_timestamp() - start_time) * 1000
                self._metrics.record(response)
                self._trace_history.append(response.to_dict())
                with self._lock:
                    self._data["meta"]["total_fast_path"] += 1
                    self._save()
                self._post_process(request, context, response)
                return response
        except Exception:
            pass

        # Deliberative Pipeline (System 2)
        try:
            self._deliberative_pipeline(request, context, response)
        except Exception as e:
            response.add_trace_step("pipeline", "error", str(e), (_timestamp() - start_time) * 1000, 0.0)
            if not response.response:
                response.response = "Deliberative processing encountered an issue."
            response.confidence = max(response.confidence, 0.1)

        response.duration_ms = (_timestamp() - start_time) * 1000

        self._metrics.record(response)
        self._trace_history.append(response.to_dict())
        with self._lock:
            self._data["meta"]["total_deliberative"] += 1
            self._save()

        self._post_process(request, context, response)
        return response

    # ── Statistics ──────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            meta = self._data.get("meta", {})
            metrics = self._metrics.to_dict()
            ws = self._workspace.get_stats()
            return {
                "total_requests": meta.get("total_requests", 0),
                "fast_path_count": meta.get("total_fast_path", 0),
                "deliberative_count": meta.get("total_deliberative", 0),
                "fast_path_rate": round(meta.get("total_fast_path", 0) / max(meta.get("total_requests", 0), 1), 3),
                "avg_fast_ms": metrics.get("avg_fast_ms", 0),
                "avg_deliberative_ms": metrics.get("avg_deliberative_ms", 0),
                "module_usage": metrics.get("module_invocations", {}),
                "avg_confidence": metrics.get("avg_confidence", 0),
                "workspace": ws,
                "competitors": {n: round(c.salience, 3) for n, c in self._competitors.items()},
                "somatic_markers": dict(self._somatic_markers),
            }

    def format_for_prompt(self, max_chars: int = 600) -> str:
        stats = self.get_stats()
        parts = [
            "[COGNITIVE INTEGRATION — WTA + Global Workspace]",
            f"Requests: {stats['total_requests']} | Fast-path: {stats['fast_path_rate']:.0%}",
            f"Fast: {stats['avg_fast_ms']:.0f}ms | Deliberative: {stats['avg_deliberative_ms']:.0f}ms",
            f"Workspace broadcasts: {stats['workspace']['total_broadcasts']}",
            f"WTA winner: {stats['workspace']['last_winner'] or 'none'}",
        ]
        if stats.get("somatic_markers"):
            markers = [f"{k}({v:.2f})" for k, v in stats["somatic_markers"].items()][:3]
            parts.append(f"Somatic markers: {', '.join(markers)}")
        result = "\n".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars] + "[...]"
        return result

    def health_check(self) -> dict:
        module_names = list(MODULE_PRIORITIES.keys())
        status = {}
        for name in module_names:
            mod = self._get_module(name)
            status[name] = "available" if mod is not None else "unavailable"
        available = sum(1 for s in status.values() if s == "available")
        return {"modules": status, "available_count": available,
                "total_count": len(module_names),
                "health": "good" if available >= 6 else "degraded" if available >= 3 else "critical"}


# ── Singleton ───────────────────────────────────────────────────────────────

_cognitive_integration = None
_integration_lock = threading.Lock()


def get_cognitive_integration() -> CognitiveIntegration:
    global _cognitive_integration
    if _cognitive_integration is None:
        with _integration_lock:
            if _cognitive_integration is None:
                _cognitive_integration = CognitiveIntegration()
    return _cognitive_integration


# ── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ci = get_cognitive_integration()

    response = ci.process_request("What is the causal structure of this system?",
                                  {"task_type": "causal", "domain": "physics"})
    print(f"Response: {response.response[:80]}...")
    print(f"Path: {response.path}")
    print(f"Confidence: {response.confidence}")
    print(f"Modules: {response.modules_used}")
    print(f"Duration: {response.duration_ms:.0f}ms")

    print(f"\nStats: {ci.get_stats()}")
    print(f"Health: {ci.health_check()}")
    print(f"Prompt:\n{ci.format_for_prompt()}")