"""Tests for the phase-registry refactor of discovery_pipeline_v2 (smells 7-13)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from discovery.discovery_pipeline_v2 import (  # noqa: E402
    PHASE_REGISTRY,
    PipelineContext,
    MODE_LEVELS,
    list_phases,
    run_discovery_pipeline,
    _apply_adversarial_verdicts,
    _filter_contradicted_predictions,
    _CASIMIR_RE,
)
from discovery.scoring_config import (  # noqa: E402
    grade_for,
    cross_validation_adjustment,
    unify_scores,
)


# ------------------------------------------------------------------
# 1. Registry integrity (fix 7 + fix 13)
# ------------------------------------------------------------------

def test_registry_all_candidate_steps_callable():
    assert len(PHASE_REGISTRY) >= 30
    for step in PHASE_REGISTRY:
        assert callable(step.fn), step.key
        assert step.min_mode in MODE_LEVELS, step.key


def test_literature_runs_first_and_gap_discovery_last():
    assert PHASE_REGISTRY[0].key == "literature"
    assert PHASE_REGISTRY[-1].key == "gap_discovery"


def test_quick_mode_keeps_foundation_phases_only():
    keys = [k for _, k, _ in list_phases("quick")]
    assert "literature" in keys and "gap_detection" in keys
    assert "missing_variables" not in keys
    assert "theory_competition" not in keys
    assert "reflexion" not in keys


def test_standard_mode_adds_analysis_phases_but_not_full():
    keys = [k for _, k, _ in list_phases("standard")]
    assert "missing_variables" in keys and "curious_questioning" in keys
    assert "mechanism_generation" not in keys
    assert "discovery_scoring" not in keys


def test_full_mode_contains_all_phases():
    quick = set(k for _, k, _ in list_phases("quick"))
    std = set(k for _, k, _ in list_phases("standard"))
    full = set(k for _, k, _ in list_phases("full"))
    assert quick <= std <= full
    assert len(full) == len(PHASE_REGISTRY)


def test_unknown_mode_falls_back_to_full():
    assert len(list_phases("banana")) == len(PHASE_REGISTRY)


# ------------------------------------------------------------------
# 2. Adversarial verdict gate (fix 9)
# ------------------------------------------------------------------

def _ctx_with(pool_names, verdicts_by_name, winner_name=None):
    ctx = PipelineContext("test topic", "space_astronomy", "full", None, "", "auto")
    ctx.theories = [{"name": n, "scores": {"overall": 1.0}} for n in pool_names]
    ctx.winner = {"name": winner_name, "scores": {"overall": 1.0}} if winner_name else None
    ctx.report["phases"]["theory_competition"] = {}
    ctx.report["phases"]["adversarial_test"] = {
        "theory_tests": [
            {"name": name, "verdict": verdict, "reasoning": "because " + verdict}
            for name, verdict in verdicts_by_name.items()
        ]
    }
    return ctx


def test_killed_theories_are_removed_from_pool():
    ctx = _ctx_with(["T1", "T2", "T3"], {"T1": "survived", "T2": "killed", "T3": "weakened"})
    _apply_adversarial_verdicts(ctx)
    names = [t["name"] for t in ctx.theories]
    assert names == ["T1", "T3"]
    assert ctx.report["phases"]["theory_competition"]["theories"] == ctx.theories


def test_elimination_metadata_is_recorded():
    ctx = _ctx_with(["T1", "T2"], {"T1": "killed", "T2": "survived"})
    _apply_adversarial_verdicts(ctx)
    assert ctx.report["phases"]["adversarial_test"]["eliminated_theories"][0]["name"] == "T1"
    assert ctx.report["phases"]["adversarial_test"]["eliminated_theories"][0]["reason"].startswith("because killed")


def test_weakened_theories_get_flagged_not_removed():
    ctx = _ctx_with(["T1"], {"T1": "weakened"})
    _apply_adversarial_verdicts(ctx)
    assert ctx.theories[0]["adversarial_verdict"] == "weakened"
    assert ctx.theories[0]["scores"]["adversarial_weakened"] is True


def test_killed_winner_is_replaced_by_best_survivor():
    ctx = _ctx_with(["W", "B"], {"W": "killed", "B": "survived"}, winner_name="W")
    ctx.theories[0]["scores"]["overall"] = 0.9
    ctx.theories[1]["scores"]["overall"] = 0.5
    _apply_adversarial_verdicts(ctx)
    assert ctx.winner["name"] == "B"
    assert ctx.report["phases"]["theory_competition"]["winner_name"] == "B"


def test_all_killed_falls_back_to_tournament_ranking():
    ctx = _ctx_with(["T1", "T2"], {"T1": "killed", "T2": "killed"})
    _apply_adversarial_verdicts(ctx)
    assert len(ctx.theories) == 2  # nothing removed
    assert ctx.report["phases"]["adversarial_test"]["elimination_note"] is not None


# ------------------------------------------------------------------
# 3. Scoring config (fix 12)
# ------------------------------------------------------------------

def test_grade_for_matches_original_thresholds():
    assert grade_for(80) == "A" and grade_for(100) == "A"
    assert grade_for(65) == "B" and grade_for(79.9) == "B"
    assert grade_for(50) == "C" and grade_for(35) == "D"
    assert grade_for(0) == "F"


def test_cv_penalty_formula_matches_original():
    # original: robustness 0.3 → penalty (0.5-0.3)*20 = 4.0
    assert cross_validation_adjustment(0.3) == {"kind": "penalty", "amount": 4.0}
    assert cross_validation_adjustment(0.5) == {"kind": "neutral", "amount": 0.0}
    # original: robustness 0.6 → bonus (0.6-0.5)*10 = 1.0
    assert cross_validation_adjustment(0.6) == {"kind": "bonus", "amount": 1.0}


def test_unify_scores_matches_original_blend():
    u = unify_scores(80, 40)
    # blend = 80*0.6 + 40*0.4 = 64, BUT original behavior caps when refinement
    # < 0.7*discovery: min(64, 40+10) = 50 — the cap is the point.
    assert u["unified_score"] == 50.0
    assert u["capped"] is True
    assert u["override_discovery"] is True
    # cleanly consistent refinement → no cap, no override
    u2 = unify_scores(60.0, 55.0)
    assert u2["capped"] is False and u2["override_discovery"] is False


# ------------------------------------------------------------------
# 4. LLM prompt truncation (fix 10)
# ------------------------------------------------------------------

def test_prompt_under_limit_untouched():
    import discovery.discovery_pipeline_v2 as dpv
    dpv.llm_call = lambda *a, **k: "OK"
    small = "x" * 1000
    assert dpv._truncated_llm(small) == "OK"


def test_big_prompt_is_truncated_to_limit_with_marker():
    import discovery.discovery_pipeline_v2 as dpv
    dpv.llm_call = lambda *a, **k: "OK"
    big = "head" + "x" * 60000 + "tail"
    sent = {}
    def capture(prompt, **kwargs):
        sent["p"] = prompt
        return "OK"
    dpv.llm_call = capture
    dpv._truncated_llm(big)
    assert "context truncated" in sent["p"]
    assert len(sent["p"]) < 60000
    assert sent["p"].startswith("head")
    assert sent["p"].endswith("tail")  # tail preserved too


# ------------------------------------------------------------------
# 5. Casimir gating (fix 8)
# ------------------------------------------------------------------

def test_casimir_detector():
    assert _CASIMIR_RE.search("gold plate force between surfaces")
    assert _CASIMIR_RE.search("Casimir pressure at 100nm")
    assert _CASIMIR_RE.search("van der Waals interaction")
    assert not _CASIMIR_RE.search("exoplanet transit spectroscopy JWST biosignatures")


def test_filter_contradicted_predictions_still_guards():
    preds = [{"statement": "X increases Y"}]
    assert _filter_contradicted_predictions(preds, []) == preds


# ------------------------------------------------------------------
# 6. API signature parity — no callers break
# ------------------------------------------------------------------

def test_run_discovery_pipeline_signature_unchanged():
    import inspect
    sig = inspect.signature(run_discovery_pipeline)
    params = list(sig.parameters)
    assert params == ["topic", "domain", "mode", "curiosity_constraint", "original_topic"]
    defaults = [p.default for p in sig.parameters.values()]
    assert defaults[1] == "" and defaults[2] == "full"
