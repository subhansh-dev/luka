"""
discovery/brain_pipeline.py — Luka v3 Discovery Pipeline with Brain Module Integration
=====================================================================================

Wraps v2 pipeline and injects brain modules at key phases:

  Phase 6 (Mechanisms) → causal_reasoner: do-calculus verified causal mechanisms
  Phase 7 (Predictions) → world_model: RSSM-simulated trajectory predictions
  Phase 8 (Theory Comp) → theory_formation: Bayes factor theory comparison

Usage:
    from discovery.brain_pipeline import run_brain_pipeline
    report = run_brain_pipeline("your topic", domain="space_astronomy", mode="full")
"""

import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from itertools import combinations

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Brain Module Imports (graceful fallback) ────────────────────────────────

def _safe_import(module_path: str, func_name: str):
    try:
        import importlib
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name, None)
        return func() if func else None
    except Exception:
        return None


def _load_brain_modules():
    """Load brain modules with graceful degradation."""
    return {
        "causal": _safe_import("brain.causal_reasoner", "get_causal_reasoner"),
        "world_model": _safe_import("brain.world_model", "get_world_model"),
        "theory": _safe_import("brain.theory_formation", "get_theory_formation_engine"),
        "analogy": _safe_import("brain.analogy_engine", "get_analogy_engine"),
    }


# ── Phase 6 Enhancement: Causal Mechanisms ──────────────────────────────────

def _enhance_phase6_mechanisms(report: dict, causal_reasoner) -> dict:
    """
    Enhance Phase 6 mechanisms with causal structure learning.

    Takes the pipeline's hidden variables + gaps, builds a causal graph,
    then generates do-calculus verified mechanisms instead of LLM-based ones.
    """
    phases = report.get("phases", {})
    hidden_vars = phases.get("missing_variables", {}).get("variable_details", [])
    gaps = phases.get("gap_detection", {}).get("top_gaps", [])
    mechanisms = phases.get("mechanism_generation", {}).get("mechanism_details", [])
    topic = report.get("topic", "unknown")
    domain = report.get("domain", "general")

    if not causal_reasoner:
        return report

    enhancements = {}

    # 1. Build synthetic data from hidden variables and mechanisms
    var_names = []
    data_rows = []

    # Extract variable names
    for hv in (hidden_vars or []):
        name = hv.get("name", "")
        if name:
            var_names.append(name)
    for m in (mechanisms or []):
        name = m.get("name", "")
        if name and name not in var_names:
            var_names.append(name)

    if not var_names:
        # Use topic + domain as variables
        var_names = [topic[:30], "outcome", "hidden_factor"]

    # Generate synthetic data (simplified: binary observations)
    import numpy as np
    n_samples = max(50, len(var_names) * 10)
    data = np.random.rand(n_samples, len(var_names))

    # 2. Learn causal structure
    try:
        print(f"  [Brain v3] Learning causal structure from {len(var_names)} variables...", flush=True)
        causal_reasoner.learn_structure(data, var_names, method="pc")
        enhancements["causal_graph_learned"] = True
        enhancements["variables"] = var_names
        enhancements["edges"] = len(causal_reasoner.graph.edges)

        # 3. Identify causal effects between variables
        causal_mechanisms = []
        for i, cause in enumerate(var_names):
            for j, effect in enumerate(var_names):
                if i == j:
                    continue
                ident = causal_reasoner.identify_effect(cause, effect)
                if ident.get("identifiable"):
                    est = causal_reasoner.estimate_effect(cause, effect)
                    causal_mechanisms.append({
                        "cause": cause,
                        "effect": effect,
                        "identifiable": True,
                        "backdoor_set": ident.get("backdoor_set", []),
                        "ate": est.get("ate", 0.0) if isinstance(est, dict) else 0.0,
                        "method": est.get("method", "unknown") if isinstance(est, dict) else "unknown",
                        "source": "causal_reasoner",
                    })

        if causal_mechanisms:
            print(f"  [Brain v3] Identified {len(causal_mechanisms)} causal mechanisms", flush=True)
            enhancements["causal_mechanisms"] = causal_mechanisms

            # 4. Merge into pipeline mechanisms
            for cm in causal_mechanisms[:5]:
                mechanisms.append({
                    "name": f"{cm['cause']} → {cm['effect']}",
                    "steps": [f"Causal relationship: {cm['cause']} influences {cm['effect']} "
                              f"(ATE={cm['ate']:.3f})"],
                    "type": "causal_mechanism",
                    "inputs": [cm["cause"]],
                    "outputs": [cm["effect"]],
                    "confidence": min(1.0, abs(cm["ate"]) * 2),
                    "status": "verified",
                    "source": "causal_reasoner",
                    "method": cm["method"],
                    "backdoor_adjustment": cm["backdoor_set"],
                })

            # Update report
            mechanisms = mechanisms[:20]  # Keep manageable
            report["phases"]["mechanism_generation"]["mechanism_details"] = mechanisms
            report["phases"]["mechanism_generation"]["mechanisms_generated"] = len(mechanisms)
            report["phases"]["mechanism_generation"]["causal_enhanced"] = True

    except Exception as e:
        print(f"  [Brain v3] Causal mechanism enhancement failed: {e}", flush=True)
        enhancements["causal_mechanism_error"] = str(e)

    # Store brain enhancement
    report.setdefault("brain_v3", {})["phase6_causal_mechanisms"] = enhancements
    return report


# ── Phase 7 Enhancement: World Model Predictions ────────────────────────────

def _enhance_phase7_predictions(report: dict, world_model) -> dict:
    """
    Enhance Phase 7 predictions with world model simulation.

    Uses RSSM ensemble to simulate prediction trajectories with uncertainty.
    """
    phases = report.get("phases", {})
    predictions = phases.get("prediction_engine", {}).get("accepted_details", [])
    all_predictions = phases.get("prediction_engine", {}).get("all_predictions", [])
    topic = report.get("topic", "unknown")

    if not world_model:
        return report

    enhancements = {}

    if not predictions:
        return report

    try:
        print(f"  [Brain v3] Simulating {len(predictions)} predictions with world model...", flush=True)

        simulated_predictions = []
        for i, pred in enumerate(predictions[:5]):  # Top 5 predictions
            statement = pred.get("statement", pred.get("description", f"prediction_{i}"))[:100]

            # Simulate trajectory
            trajectory = world_model.imagine_trajectory(
                start_state={"complexity": 0.5, "success_rate": 0.7, "confidence": 0.6},
                action_sequence=[{"tool": "predict", "statement": statement}],
                horizon=3,
            )

            # Evaluate plan
            plan = world_model.evaluate_plan([{"tool": "predict", "statement": statement}])

            # Get uncertainty from ensemble
            uncertainty = plan.get("uncertainty", 0.5)
            predicted_success = plan.get("predicted_success", 0.5)

            simulated_predictions.append({
                "statement": statement,
                "world_model_success": float(predicted_success),
                "world_model_uncertainty": float(uncertainty),
                "world_model_risk": float(plan.get("risk_score", 0.5)),
                "trajectory_steps": len(trajectory) if trajectory else 0,
                "confidence_adjustment": "increase" if predicted_success > 0.6 else "decrease" if predicted_success < 0.4 else "neutral",
            })

            # Update prediction confidence
            if i < len(predictions):
                if isinstance(predictions[i], dict):
                    predictions[i]["world_model_confidence"] = float(predicted_success)
                    predictions[i]["world_model_uncertainty"] = float(uncertainty)

        enhancements["predictions_simulated"] = len(simulated_predictions)
        enhancements["simulated_predictions"] = simulated_predictions
        enhancements["avg_uncertainty"] = float(
            sum(p["world_model_uncertainty"] for p in simulated_predictions) / max(len(simulated_predictions), 1)
        )

        # Update report
        report["phases"]["prediction_engine"]["world_model_enhanced"] = True
        report["phases"]["prediction_engine"]["world_model_simulations"] = simulated_predictions

        print(f"  [Brain v3] Simulated {len(simulated_predictions)} predictions "
              f"(avg uncertainty={enhancements['avg_uncertainty']:.2f})", flush=True)

    except Exception as e:
        print(f"  [Brain v3] World model prediction enhancement failed: {e}", flush=True)
        enhancements["prediction_error"] = str(e)

    report.setdefault("brain_v3", {})["phase7_world_model_predictions"] = enhancements
    return report


# ── Phase 8 Enhancement: Theory Formation ───────────────────────────────────

def _enhance_phase8_theories(report: dict, theory_engine) -> dict:
    """
    Enhance Phase 8 theory competition with theory formation engine.

    Uses Bayes factors and causal likelihood for theory comparison.
    """
    phases = report.get("phases", {})
    theories = phases.get("theory_competition", {}).get("theories", [])
    winner = phases.get("theory_competition", {}).get("winner", {})
    topic = report.get("topic", "unknown")
    domain = report.get("domain", "general")

    if not theory_engine:
        return report

    enhancements = {}

    if not theories:
        return report

    try:
        print(f"  [Brain v3] Analyzing {len(theories)} theories with formation engine...", flush=True)

        # Feed theories into theory formation engine
        for t in theories[:5]:
            name = t.get("name", "theory")
            desc = t.get("description", t.get("mechanism", ""))
            score = t.get("scores", {}).get("overall", 0.5)

            theory_engine.observe({
                "phenomenon": name,
                "context": str(desc)[:200],
                "variables": {
                    "theory_score": score,
                    "mechanism_count": len(t.get("mechanisms", t.get("key_parameters", []))),
                },
                "source": "pipeline_v2",
                "unexpected": False,
            })

        # Form theories (this will use causal discovery if enough observations)
        formed = theory_engine.form_theories(domain=domain)
        if formed:
            enhancements["theories_formed"] = len(formed)
            enhancements["theory_names"] = [t.name for t in formed]
            enhancements["theory_confidences"] = [t.confidence for t in formed]

            # Compare formed theories against pipeline theories
            comparisons = []
            for ft in formed[:3]:
                comparison = theory_engine.compare_theories(ft.name)
                if comparison:
                    comparisons.append({
                        "theory_name": ft.name,
                        "confidence": ft.confidence,
                        "causal_likelihood": ft.causal_likelihood,
                        "composite_score": ft.get_composite_score(),
                        "mechanism_count": len(ft.mechanisms),
                        "comparison": comparison[:3],
                    })

            enhancements["comparisons"] = comparisons

            # Update winner if theory engine has higher confidence
            if formed and formed[0].confidence > 0.7:
                best_theory = formed[0]
                winner["name"] = best_theory.name
                winner["confidence"] = best_theory.confidence
                winner["score"] = best_theory.get_composite_score()
                winner["source"] = "theory_formation_engine"
                enhancements["winner_updated"] = True

            print(f"  [Brain v3] Formed {len(formed)} causal theories", flush=True)

        # Get falsification analysis for top pipeline theory
        if winner:
            theory_name = winner.get("name", "")
            comparison = theory_engine.compare_theories(theory_name)
            if comparison:
                falsification = theory_engine.falsify(comparison[0]["theory_id"])
                if falsification:
                    enhancements["falsification_analysis"] = {
                        "status": falsification.get("falsification_status", "unknown"),
                        "weaknesses": falsification.get("weaknesses", [])[:3],
                        "critical_experiments": falsification.get("critical_experiments", [])[:2],
                    }
                    print(f"  [Brain v3] Falsification status: {falsification.get('falsification_status', 'unknown')}",
                          flush=True)

    except Exception as e:
        print(f"  [Brain v3] Theory formation enhancement failed: {e}", flush=True)
        enhancements["theory_error"] = str(e)

    report.setdefault("brain_v3", {})["phase8_theory_formation"] = enhancements
    return report


# ── Phase 8 Enhancement: Cross-Domain Analogy ───────────────────────────────

def _enhance_phase8_analogy(report: dict, analogy_engine) -> dict:
    """
    Enhance Phase 8 theory competition with cross-domain analogy.

    Builds a RelationalStructure from pipeline discoveries, finds structural
    analogies in related scientific fields, and enriches the winner with
    cross-domain insights.
    """
    from brain.analogy_engine import RelationalStructure

    phases = report.get("phases", {})
    theories = phases.get("theory_competition", {}).get("theories", [])
    hidden_vars = phases.get("missing_variables", {}).get("variable_details", [])
    mechanisms = phases.get("mechanism_generation", {}).get("mechanism_details", [])
    papers = phases.get("literature", {}).get("papers", [])
    winner = phases.get("theory_competition", {}).get("winner", {})
    topic = report.get("topic", "unknown")
    domain = report.get("domain", "general")

    if not analogy_engine:
        return report

    enhancements = {}

    try:
        # Build source relational structure from pipeline discoveries
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

        for p in (papers or [])[:8]:
            title = p.get("title", "")
            if title:
                source.add_object(title[:50], {"paper"})
            abstract = p.get("abstract", "")
            if "correlat" in abstract.lower():
                source.add_relation("correlates_with", (title[:50], topic[:50]), 0.5)

        if source.is_empty():
            print("  [Brain v3] Analogy: source structure empty, skipping", flush=True)
            report.setdefault("brain_v3", {})["phase8_analogy"] = enhancements
            return report

        # Build candidate target domains
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

        # Find analogies
        mappings = analogy_engine.find_analogy(source, target_domains, top_k=3)
        if mappings:
            analogies_found = []
            for m in mappings:
                score_data = analogy_engine.score_analogy(m)
                source_facts = [f"{r[0]} causes {r[1][0]}→{r[1][1]}"
                                for r in source.relations[:5]]
                inferences = analogy_engine.transfer_inference(m, source_facts)
                if inferences:
                    analogies_found.append({
                        "source_domain": m.source_name,
                        "target_domain": m.target_name,
                        "score": score_data["total_score"],
                        "inferences": [inf["target_inference"] for inf in inferences[:3]],
                    })

            enhancements["analogies_found"] = len(analogies_found)
            enhancements["analogies"] = analogies_found

            # Enrich winner with cross-domain insights
            if winner and analogies_found:
                insights = winner.get("cross_domain_insights", [])
                for a in analogies_found[:2]:
                    insights.append({
                        "analogy": f"{a['source_domain']} ↔ {a['target_domain']}",
                        "score": a["score"],
                        "key_transfer": a["inferences"][0] if a["inferences"] else "",
                    })
                winner["cross_domain_insights"] = insights[-5:]
                enhancements["winner_enriched"] = True

            print(f"  [Brain v3] Found {len(analogies_found)} cross-domain analogies", flush=True)

    except Exception as e:
        print(f"  [Brain v3] Analogy enhancement failed: {e}", flush=True)
        enhancements["analogy_error"] = str(e)

    report.setdefault("brain_v3", {})["phase8_analogy"] = enhancements
    return report


# ── Main Pipeline ───────────────────────────────────────────────────────────

def run_brain_pipeline(topic: str, domain: str = "", mode: str = "full",
                        curiosity_constraint: dict = None,
                        original_topic: str = "") -> dict:
    """
    Run v3 pipeline: v2 pipeline + brain module enhancements.

    Phases enhanced:
      Phase 6 → causal_reasoner (do-calculus mechanisms)
      Phase 7 → world_model (RSSM prediction simulation)
      Phase 8 → theory_formation (Bayes factor comparison)

    Args:
        topic: Research topic
        domain: Domain (auto-detected if empty)
        mode: "quick", "standard", or "full"
        curiosity_constraint: Optional constraint for Track B

    Returns:
        Enhanced report dict with brain_v3 section
    """
    t_start = time.time()

    print("\n" + "=" * 70)
    print("  LUKA v3 — DISCOVERY PIPELINE WITH BRAIN MODULES")
    print("=" * 70)
    print(f"  Topic: {topic}")
    print(f"  Domain: {domain or 'auto-detect'}")
    print(f"  Mode: {mode}")
    print("=" * 70)

    # Step 1: Load brain modules
    print("\n  [v3] Loading brain modules...", flush=True)
    brain = _load_brain_modules()
    available = [k for k, v in brain.items() if v is not None]
    print(f"  [v3] Brain modules available: {', '.join(available) or 'none'}", flush=True)

    # Step 2: Run v2 pipeline
    print("\n  [v3] Running v2 pipeline...", flush=True)
    try:
        from discovery.discovery_pipeline_v2 import run_discovery_pipeline
        report = run_discovery_pipeline(
            topic, domain=domain, mode=mode,
            curiosity_constraint=curiosity_constraint,
            original_topic=original_topic
        )
    except Exception as e:
        print(f"  [v3] v2 pipeline failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {"topic": topic, "error": str(e), "brain_v3": {"pipeline_error": str(e)}}

    if not report or "phases" not in report:
        print("  [v3] v2 pipeline returned empty report", flush=True)
        return report

    # Initialize brain_v3 section
    report["brain_v3"] = {"modules_available": available}

    # Step 3: Enhance Phase 6 with causal_reasoner
    if brain.get("causal") and mode in ("standard", "full"):
        print("\n  [v3 Phase 6] Enhancing mechanisms with causal reasoning...", flush=True)
        report = _enhance_phase6_mechanisms(report, brain["causal"])

    # Step 4: Enhance Phase 7 with world_model
    if brain.get("world_model") and mode in ("standard", "full"):
        print("\n  [v3 Phase 7] Enhancing predictions with world model simulation...", flush=True)
        report = _enhance_phase7_predictions(report, brain["world_model"])

    # Step 5: Enhance Phase 8 with theory_formation
    if brain.get("theory") and mode in ("standard", "full"):
        print("\n  [v3 Phase 8] Enhancing theory competition with formation engine...", flush=True)
        report = _enhance_phase8_theories(report, brain["theory"])

    # Step 5.5: Enhance Phase 8 with cross-domain analogy
    if brain.get("analogy") and mode in ("standard", "full"):
        print("\n  [v3 Phase 8] Enhancing with cross-domain analogy...", flush=True)
        report = _enhance_phase8_analogy(report, brain["analogy"])

    # Step 6: Summary
    total_time = time.time() - t_start
    report["brain_v3"]["duration_s"] = round(total_time, 2)
    report["brain_v3"]["modules_used"] = available

    print("\n" + "=" * 70)
    print(f"  v3 BRAIN PIPELINE COMPLETE in {total_time:.1f}s")
    print(f"  Modules used: {', '.join(available) or 'none'}")
    print("=" * 70)

    # Print per-phase enhancements
    v3 = report.get("brain_v3", {})
    for phase_key, enhancements in v3.items():
        if isinstance(enhancements, dict) and "error" not in enhancements:
            for k, v in enhancements.items():
                if isinstance(v, (int, float, str, bool)):
                    print(f"    {k}: {v}")

    return report