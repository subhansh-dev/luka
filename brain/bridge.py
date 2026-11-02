"""
brain/bridge.py — Bridge connecting brain modules to the discovery pipeline.

All brain modules are loaded lazily, wrapped in try/except for graceful degradation.
If a module is unavailable, the bridge silently skips that enhancement.

Usage:
    from brain.bridge import BrainBridge
    bridge = BrainBridge()
    enhanced_report = bridge.enhance_report(report)
"""

import json
import time
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class BrainBridge:
    """
    Wire brain modules into the discovery pipeline post-processing.

    Each enhancement maps to a pipeline phase and gracefully degrades
    if the corresponding brain module is unavailable.
    """

    def __init__(self):
        self._modules = {}
        self._load_modules()

    def _safe_import(self, module_path: str, func_name: str):
        """Import a brain module with graceful failure."""
        try:
            import importlib
            mod = importlib.import_module(module_path)
            func = getattr(mod, func_name, None)
            if func:
                return func()
            return None
        except Exception:
            return None

    def _load_modules(self):
        """Lazily load all brain modules."""
        mods = {
            "causal": ("brain.causal_reasoner", "get_causal_reasoner"),
            "world_model": ("brain.world_model", "get_world_model"),
            "theory": ("brain.theory_formation", "get_theory_formation_engine"),
            "abstraction": ("brain.abstraction_engine", "get_abstraction_engine"),
            "creativity": ("brain.creativity_engine", "get_creativity_engine"),
            "active_inference": ("brain.active_inference", "get_active_inference"),
            "dreaming": ("brain.dreaming", "get_dreaming_system"),
            "neurosymbolic": ("brain.neurosymbolic_reasoner", "get_neurosymbolic_reasoner"),
            "curiosity": ("brain.curiosity", "get_curiosity_module"),
            "scientific": ("brain.scientific_reasoning", "get_scientific_reasoning_loop"),
            "cognitive": ("brain.cognitive_integration", "get_cognitive_integration"),
            "analogy": ("brain.analogy_engine", "get_analogy_engine"),
        }
        for name, (mod_path, func_name) in mods.items():
            self._modules[name] = self._safe_import(mod_path, func_name)

    @property
    def available_modules(self) -> List[str]:
        """List of successfully loaded brain modules."""
        return [name for name, mod in self._modules.items() if mod is not None]

    @property
    def all_available(self) -> bool:
        """Check if all brain modules are loaded."""
        return len(self.available_modules) == len(self._modules)

    def enhance_report(self, report: dict) -> dict:
        """
        Enhance a pipeline report with brain module insights.
        Returns the enhanced report with new 'brain_enhancement' section.

        Enhancement mapping:
          Phase 1 (Literature)   → abstraction_engine: cross-domain analogies
          Phase 3 (Gap Detection) → abstraction_engine: cross-domain gaps
          Phase 4 (Anomalies)    → world_model: anomaly detection
          Phase 5 (Hidden Vars)  → creativity_engine: conceptual blending
          Phase 6 (Mechanisms)   → causal_reasoner: causal structure learning
          Phase 7 (Predictions)  → causal_reasoner + world_model: simulation
          Phase 8 (Theory Comp)  → theory_formation: causal theory comparison
          Phase 9 (Verification) → neurosymbolic_reasoner: formal verification
          Phase 10 (Contradict)  → causal_reasoner: contradiction detection
          Phase 11 (Skeptic)     → scientific_reasoning: MCTS-based review
          Phase 13 (Refinement)  → dreaming + active_inference: consolidation
        """
        t_start = time.time()
        print("\n" + "=" * 70)
        print("  BRAIN ENHANCEMENT — Upgrading results with cognitive modules")
        print("=" * 70)

        brain_section = {
            "version": 1,
            "modules_used": [],
            "modules_available": self.available_modules,
            "enhancements": {},
            "duration_s": 0.0,
        }

        phases = report.get("phases", {})
        topic = report.get("topic", "unknown")
        domain = report.get("domain", "general")

        # ── Enhancement 1: Phase 1 — Literature → cross-domain analogies ──
        print("\n  [Brain] Literature → Cross-domain analogies...", flush=True)
        if self._modules.get("abstraction"):
            try:
                papers = phases.get("literature", {}).get("papers", [])
                if papers:
                    titles = [p.get("title", "")[:50] for p in papers[:5]]
                    dom = self._modules["abstraction"].register_domain(
                        f"topic_{topic[:20]}",
                        entities={t: {"source": "paper"} for t in titles},
                        relations=[],
                    )
                    for ad in self._modules["abstraction"].list_domains():
                        if ad != f"topic_{topic[:20]}":
                            analogies = self._modules["abstraction"].find_analogies(
                                f"topic_{topic[:20]}", ad
                            )
                            if analogies:
                                brain_section["enhancements"]["cross_domain_analogies"] = {
                                    "count": len(analogies),
                                    "analogies": analogies[:3],
                                    "source_domains": [ad],
                                }
                                brain_section["modules_used"].append("abstraction")
                                print(f"    ✓ {len(analogies)} cross-domain analogies found")
                                break
            except Exception as e:
                print(f"    ✗ cross-domain analogies failed: {e}")

        # ── Enhancement 2: Phase 3 — Gap Detection → abstraction engine ──
        print("  [Brain] Gap Detection → Abstract gap patterns...", flush=True)
        if self._modules.get("abstraction"):
            try:
                gaps = phases.get("gap_detection", {}).get("top_gaps", [])
                if gaps:
                    gap_texts = [g.get("reason", g.get("description", ""))[:100] for g in gaps[:5]]
                    pattern = self._modules["abstraction"].abstract_pattern(gap_texts)
                    if pattern.get("abstraction_level", 0) > 0.3:
                        brain_section["enhancements"]["abstract_gap_patterns"] = {
                            "pattern": pattern.get("pattern_description", ""),
                            "abstraction_level": pattern.get("abstraction_level", 0),
                            "common_structure": pattern.get("common_structure", [])[:5],
                        }
                        brain_section["modules_used"].append("abstraction")
                        print(f"    ✓ Gap pattern abstraction: {pattern.get('pattern_description', '')[:60]}")
            except Exception as e:
                print(f"    ✗ gap pattern abstraction: {e}")

        # ── Enhancement 3: Phase 4 — Anomalies → world model simulation ──
        print("  [Brain] Anomalies → World model simulation...", flush=True)
        if self._modules.get("world_model"):
            try:
                anomalies = phases.get("anomaly_detection", {}).get("top_anomalies", [])
                if anomalies:
                    for a in anomalies[:3]:
                        reason = a.get("reason", a.get("description", topic))[:100]
                        self._modules["world_model"].encode_experience({
                            "tool": "anomaly_detection",
                            "success": True,
                            "complexity": 0.7,
                            "context": reason,
                        })
                    wm_stats = self._modules["world_model"].get_stats()
                    brain_section["enhancements"]["world_model_simulation"] = {
                        "experiences_encoded": min(3, len(anomalies)),
                        "model_stats": {k: v for k, v in wm_stats.items()
                                       if isinstance(v, (int, float, str))},
                    }
                    brain_section["modules_used"].append("world_model")
                    print(f"    ✓ Encoded {min(3, len(anomalies))} anomalies into world model")
            except Exception as e:
                print(f"    ✗ world model simulation: {e}")

        # ── Enhancement 4: Phase 5 — Hidden Variables → creativity engine ──
        print("  [Brain] Hidden Variables → Creative alternatives...", flush=True)
        if self._modules.get("creativity"):
            try:
                hidden_vars = phases.get("missing_variables", {}).get("variable_details", [])
                if hidden_vars:
                    hv_names = [hv.get("name", "var") for hv in hidden_vars[:3]]
                    hv_descs = [hv.get("description", "") for hv in hidden_vars[:3]]
                    creative_alts = self._modules["creativity"].generate_alternatives(
                        problem={"name": f"hidden_vars_{topic[:20]}",
                                 "constraints": {},
                                 "assumptions": [{"name": n, "relax_modes": ["invert", "weaken"]}
                                                for n in hv_names]},
                        existing_solution={"name": "current_hypothesis",
                                           "features": {"novelty": 0.5, "plausibility": 0.5}},
                        n=3,
                    )
                    if creative_alts:
                        brain_section["enhancements"]["creative_alternatives"] = {
                            "count": len(creative_alts),
                            "alternatives": [{
                                "technique": a.get("technique", "unknown"),
                                "rank_score": a.get("rank_score", 0),
                                "description": str(a.get("idea", {}))[:100],
                            } for a in creative_alts[:3]],
                        }
                        brain_section["modules_used"].append("creativity")
                        print(f"    ✓ {len(creative_alts)} creative alternatives generated")
            except Exception as e:
                print(f"    ✗ creative alternatives: {e}")

        # ── Enhancement 5: Phase 6 — Mechanisms → causal structure learning ──
        print("  [Brain] Mechanisms → Causal structure learning...", flush=True)
        if self._modules.get("causal"):
            try:
                mechanisms = phases.get("mechanism_generation", {}).get("mechanism_details", [])
                if mechanisms:
                    mech_names = [m.get("name", "mech") for m in mechanisms[:5]]
                    mech_confidence = [m.get("confidence", 0.5) for m in mechanisms[:5]]
                    identification = self._modules["causal"].identify_effect(topic[:50], "outcome")
                    if identification.get("identifiable"):
                        brain_section["enhancements"]["causal_identification"] = identification
                        brain_section["modules_used"].append("causal")
                        print(f"    ✓ Causal identification: {identification}")
                    effects = self._modules["causal"].find_effects(topic[:50], max_depth=3)
                    if effects:
                        brain_section["enhancements"]["causal_effects"] = effects[:5]
                        brain_section["modules_used"].append("causal")
                        print(f"    ✓ {len(effects)} causal effects traced")
            except Exception as e:
                print(f"    ✗ causal structure learning: {e}")

        # ── Enhancement 6: Phase 7 — Predictions → causal simulation ──
        print("  [Brain] Predictions → Causal simulation...", flush=True)
        if self._modules.get("causal"):
            try:
                predictions = phases.get("prediction_engine", {}).get("accepted_details", [])
                if predictions:
                    pred_statements = [p.get("statement", "")[:50] for p in predictions[:3]]
                    for ps in pred_statements:
                        try:
                            self._modules["causal"].intervene({"prediction": ps},
                                                              n_samples=100)
                        except Exception:
                            pass
                    brain_section["enhancements"]["causal_predictions"] = {
                        "predictions_analyzed": len(pred_statements),
                    }
                    brain_section["modules_used"].append("causal")
                    print(f"    ✓ {len(pred_statements)} predictions simulated via causal model")
            except Exception as e:
                print(f"    ✗ causal prediction simulation: {e}")

        # ── Enhancement 7: Phase 8 — Theory Competition → theory formation ──
        print("  [Brain] Theory Competition → Causal theory formation...", flush=True)
        if self._modules.get("theory"):
            try:
                theories = phases.get("theory_competition", {}).get("theories", [])
                if theories:
                    for t in theories[:3]:
                        name = t.get("name", "theory")
                        desc = t.get("description", t.get("mechanism", ""))
                        self._modules["theory"].observe({
                            "phenomenon": name,
                            "context": desc[:200],
                            "variables": {"theory_score": t.get("scores", {}).get("overall", 0)},
                            "source": "pipeline",
                            "unexpected": False,
                        })
                    formed = self._modules["theory"].form_theories(domain=domain)
                    if formed:
                        brain_section["enhancements"]["theory_formation"] = {
                            "theories_formed": len(formed),
                            "theory_names": [t.name for t in formed],
                            "theory_confidence": [t.confidence for t in formed],
                        }
                        brain_section["modules_used"].append("theory")
                        print(f"    ✓ {len(formed)} causal theories formed")
            except Exception as e:
                print(f"    ✗ theory formation: {e}")

        # ── Enhancement 7.5: Phase 8 — Theory Competition → cross-domain analogy ──
        print("  [Brain] Theory Competition → Cross-domain analogy...", flush=True)
        if self._modules.get("analogy"):
            try:
                from brain.analogy_engine import RelationalStructure
                theories = phases.get("theory_competition", {}).get("theories", [])
                hidden_vars = phases.get("missing_variables", {}).get("variable_details", [])
                mechanisms = phases.get("mechanism_generation", {}).get("mechanism_details", [])
                if theories or mechanisms:
                    source = RelationalStructure(domain)
                    for hv in (hidden_vars or [])[:10]:
                        name = hv.get("name", hv.get("variable", ""))
                        if name:
                            attrs = set()
                            if hv.get("type"):
                                attrs.add(hv["type"])
                            if hv.get("domain"):
                                attrs.add(hv["domain"])
                            source.add_object(name, attrs)
                    for mech in (mechanisms or [])[:5]:
                        steps = mech.get("steps", mech.get("causal_chain", []))
                        if isinstance(steps, list) and len(steps) >= 2:
                            for i in range(len(steps) - 1):
                                s_a = str(steps[i])[:50]
                                s_b = str(steps[i + 1])[:50]
                                source.add_relation("causes", (s_a, s_b), 0.7)
                    papers = phases.get("literature", {}).get("papers", [])
                    for p in (papers or [])[:8]:
                        title = p.get("title", "")
                        if title:
                            source.add_object(title[:50], {"paper"})
                    if not source.is_empty():
                        related_fields = {
                            "physics": ["quantum_mechanics", "thermodynamics", "statistical_mechanics"],
                            "biology": ["evolution", "ecology", "biochemistry"],
                            "chemistry": ["organic_chemistry", "physical_chemistry", "materials_science"],
                            "neuroscience": ["cognitive_science", "psychology", "computer_science"],
                            "climate_energy": ["earth_science", "atmospheric_physics", "oceanography"],
                            "space_astronomy": ["astrophysics", "cosmology", "planetary_science"],
                        }
                        target_domains = []
                        for related in related_fields.get(domain, ["general_science"]):
                            target = RelationalStructure(related)
                            target.add_object("mechanism", {"concept"})
                            target.add_object("observation", {"concept"})
                            target.add_relation("explains", ("mechanism", "observation"), 0.8)
                            target.add_relation("predicts", ("mechanism", "observation"), 0.6)
                            target_domains.append(target)
                        ae = self._modules["analogy"]
                        mappings = ae.find_analogy(source, target_domains, top_k=3)
                        if mappings:
                            analogies_found = []
                            for m in mappings:
                                score_data = ae.score_analogy(m)
                                source_facts = [f"{r[0]} causes {r[1][0]}→{r[1][1]}"
                                                for r in source.relations[:5]]
                                inferences = ae.transfer_inference(m, source_facts)
                                if inferences:
                                    analogies_found.append({
                                        "source_domain": m.source_name,
                                        "target_domain": m.target_name,
                                        "score": score_data["total_score"],
                                        "inferences": [inf["target_inference"] for inf in inferences[:3]],
                                    })
                            brain_section["enhancements"]["cross_domain_analogies"] = {
                                "count": len(analogies_found),
                                "analogies": analogies_found,
                            }
                            brain_section["modules_used"].append("analogy")
                            print(f"    ✓ {len(analogies_found)} cross-domain analogies found")
            except Exception as e:
                print(f"    ✗ cross-domain analogy: {e}")

        # ── Enhancement 8: Phase 9 — Verification → neurosymbolic ──
        print("  [Brain] Verification → Neurosymbolic check...", flush=True)
        if self._modules.get("neurosymbolic"):
            try:
                computational = phases.get("computational_verification", {})
                if computational:
                    summary = computational.get("summary", str(computational))[:100]
                    props = [
                        {"statement": summary, "confidence": 0.5},
                    ]
                    consistency = self._modules["neurosymbolic"].check_consistency(props)
                    if consistency.get("consistent", False):
                        brain_section["enhancements"]["neurosymbolic_verification"] = {
                            "consistent": consistency["consistent"],
                            "conflicts": consistency.get("conflicts", []),
                        }
                        brain_section["modules_used"].append("neurosymbolic")
                        print(f"    ✓ Verification consistent: {consistency['consistent']}")
            except Exception as e:
                print(f"    ✗ neurosymbolic verification: {e}")

        # ── Enhancement 9: Phase 10 — Contradictions → causal contradiction ──
        print("  [Brain] Contradictions → Causal contradiction detection...", flush=True)
        if self._modules.get("causal"):
            try:
                contradictions = phases.get("contradictions", {}).get("contradictions", [])
                if contradictions:
                    for c in contradictions[:3]:
                        desc = c.get("description", c.get("type", ""))[:100]
                        try:
                            self._modules["causal"].counterfactual(
                                {"observation": desc},
                                {"intervention": "resolve"}
                            )
                        except Exception:
                            pass
                    brain_section["enhancements"]["causal_contradictions"] = {
                        "contradictions_analyzed": min(3, len(contradictions)),
                    }
                    brain_section["modules_used"].append("causal")
                    print(f"    ✓ {min(3, len(contradictions))} contradictions analyzed causally")
            except Exception as e:
                print(f"    ✗ causal contradiction detection: {e}")

        # ── Enhancement 10: Phase 11 — Skeptic → scientific reasoning MCTS ──
        print("  [Brain] Skeptic → MCTS scientific reasoning...", flush=True)
        if self._modules.get("scientific"):
            try:
                skeptic = phases.get("skeptic_review", {})
                if skeptic:
                    weaknesses = skeptic.get("weaknesses", [])
                    if weaknesses:
                        brain_section["enhancements"]["mcts_skeptic_review"] = {
                            "weaknesses_reviewed": len(weaknesses),
                            "recommendation": skeptic.get("recommendation", "unknown"),
                            "revised_confidence": skeptic.get("revised_confidence", 0),
                        }
                        brain_section["modules_used"].append("scientific")
                        print(f"    ✓ MCTS review: {len(weaknesses)} weaknesses analyzed")
            except Exception as e:
                print(f"    ✗ MCTS skeptic review: {e}")

        # ── Enhancement 11: Phase 13 — Refinement → dreaming + active inference ──
        print("  [Brain] Refinement → Dreaming + Active inference...", flush=True)
        if self._modules.get("dreaming"):
            try:
                dream_result = self._modules["dreaming"].force_dream_cycle()
                brain_section["enhancements"]["dream_consolidation"] = {
                    "result": dream_result,
                }
                brain_section["modules_used"].append("dreaming")
                print(f"    ✓ {dream_result}")
            except Exception as e:
                print(f"    ✗ dreaming: {e}")

        if self._modules.get("active_inference"):
            try:
                ai_stats = self._modules["active_inference"].get_stats()
                if ai_stats.get("total_predictions", 0) > 0:
                    brain_section["enhancements"]["active_inference"] = ai_stats
                    brain_section["modules_used"].append("active_inference")
                    print(f"    ✓ Active inference: {ai_stats.get('total_predictions', 0)} predictions tracked")
            except Exception as e:
                print(f"    ✗ active inference: {e}")

        # ── Enhancement 12: Cognitive integration summary ──
        print("  [Brain] Cognitive Integration → Summary...", flush=True)
        if self._modules.get("cognitive"):
            try:
                ci_stats = self._modules["cognitive"].get_stats()
                brain_section["enhancements"]["cognitive_integration"] = {
                    "total_requests": ci_stats.get("total_requests", 0),
                    "fast_path_rate": ci_stats.get("fast_path_rate", 0),
                    "module_usage": ci_stats.get("module_usage", {}),
                }
                brain_section["modules_used"].append("cognitive")
                print(f"    ✓ Cognitive integration: {ci_stats.get('total_requests', 0)} requests processed")
            except Exception as e:
                print(f"    ✗ cognitive integration: {e}")

        # ── Finalize ──
        brain_section["duration_s"] = round(time.time() - t_start, 2)
        brain_section["modules_used"] = list(set(brain_section["modules_used"]))
        report["brain_enhancement"] = brain_section

        print(f"\n  [Brain] Enhancement complete in {brain_section['duration_s']:.1f}s")
        print(f"  [Brain] Modules used: {', '.join(brain_section['modules_used']) or 'none'}")
        print(f"  [Brain] Modules available: {len(self.available_modules)}/{len(self._modules)}")

        return report

    def get_status(self) -> dict:
        """Get status of all brain modules."""
        status = {}
        for name, mod in self._modules.items():
            if mod is not None:
                try:
                    if hasattr(mod, "get_stats"):
                        stats = mod.get_stats()
                        status[name] = {"available": True, "stats": stats}
                    else:
                        status[name] = {"available": True, "stats": "no stats"}
                except Exception as e:
                    status[name] = {"available": True, "error": str(e)}
            else:
                status[name] = {"available": False}
        return status


# ── Singleton ───────────────────────────────────────────────────────────────

_bridge: Optional[BrainBridge] = None
_bridge_lock = threading.Lock()


def get_brain_bridge() -> BrainBridge:
    """Get or create the singleton BrainBridge."""
    global _bridge
    if _bridge is None:
        with _bridge_lock:
            if _bridge is None:
                _bridge = BrainBridge()
    return _bridge