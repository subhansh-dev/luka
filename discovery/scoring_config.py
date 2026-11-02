"""
scoring_config.py — Single source of truth for every scoring blend, weight,
and threshold magic number used by the discovery pipeline.

Rules:
  1. Any number that blends two scores, weights a dimension, or turns a
     score into a grade MUST live here, with a name and a comment explaining
     what it means and where it was calibrated from.
  2. Pipeline / tournament / runner code imports constants from this module
     instead of hardcoding them inline, so tuning happens in one place.
  3. Nothing here is allowed to import anything else from discovery/ —
     this module must stay dependency-free (pure constants + pure functions).
"""

import os

# ════════════════════════════════════════════════════════════════════
# GRADE SCALE — shared by discovery scoring, cross-validation, and
# score unification. Previously duplicated inline in three places.
# ════════════════════════════════════════════════════════════════════
# (grade, minimum_score) sorted descending. First match wins.
GRADE_THRESHOLDS: tuple = (
    ("A", 80.0),
    ("B", 65.0),
    ("C", 50.0),
    ("D", 35.0),
    ("F", 0.0),
)


def grade_for(score: float) -> str:
    """Map a 0-100 score to a letter grade using GRADE_THRESHOLDS."""
    for grade, minimum in GRADE_THRESHOLDS:
        if score >= minimum:
            return grade
    return "F"


# ════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION ADJUSTMENTS (pipeline Phase 12.5)
# How the holdout-paper robustness score skews the discovery score.
# Previously: inline `robustness < 0.5 → penalty up to 10` / bonus ≤ 5.
# Interpretation: papers=0.5 neutral; each 0.1 below threshold costs
# CV_ROBUSTNESS_PENALTY_SCALE / 10 points; above it earns bonus points.
# ════════════════════════════════════════════════════════════════════
CV_ROBUSTNESS_THRESHOLD = 0.50   # robustness below this = penalty zone
CV_ROBUSTNESS_PENALTY_SCALE = 20.0   # points deducted per 1.0 below threshold (max 10)
CV_ROBUSTNESS_BONUS_SCALE = 10.0     # points added per 1.0 above threshold (max 5)


def cross_validation_adjustment(robustness: float) -> dict:
    """Return {'kind': 'penalty'|'bonus'|'neutral', 'amount': points(≥0), ...}.

    - penalty: amount = (threshold - robustness) * PENALTY_SCALE
    - bonus:   amount = (robustness - threshold) * BONUS_SCALE
    """
    if robustness < CV_ROBUSTNESS_THRESHOLD:
        return {
            "kind": "penalty",
            "amount": round((CV_ROBUSTNESS_THRESHOLD - robustness) * CV_ROBUSTNESS_PENALTY_SCALE, 1),
        }
    if robustness > CV_ROBUSTNESS_THRESHOLD:
        return {
            "kind": "bonus",
            "amount": round((robustness - CV_ROBUSTNESS_THRESHOLD) * CV_ROBUSTNESS_BONUS_SCALE, 1),
        }
    return {"kind": "neutral", "amount": 0.0}


# ════════════════════════════════════════════════════════════════════
# SCORE UNIFICATION (pipeline Phase 13 — refinement vs discovery score)
# Final grade should respect the "Scientific Courtroom" verdict instead
# of letting the enthusiastic discovery score dominate blindly.
# Previously inlined as: unified = 0.6*discovery + 0.4*refinement, cap if
# refinement < 0.7*discovery, override if unified < 0.85*discovery, etc.
# ════════════════════════════════════════════════════════════════════
UNIFIED_DISCOVERY_WEIGHT = 0.60   # trust share of the pipeline's own scorer
UNIFIED_REFINEMENT_WEIGHT = 0.40  # trust share of the independent refinement pass

# If the refinement score is dramatically lower than the discovery score,
# the discovery scorer probably over-trusted its own output. Cap it.
UNIFIED_FLAG_RATIO = 0.70        # refinement < FLAG_RATIO × discovery = red flag
UNIFIED_FLAG_CAP_MARGIN = 10.0   # unified score capped to refinement + this margin

# Provider-bias guard: if refinement only slightly disagrees, keep the
# discovery score; big disagreements must pull the final score down.
UNIFIED_OVERRIDE_RATIO = 0.85    # unified < OVERRIDE_RATIO × discovery → override

# Below this refinement grade the refinement weakness is surfaced to the report.
UNIFIED_WEAKNESS_REFINEMENT_SCORE = 60.0


def unify_scores(discovery_score: float, refinement_score: float) -> dict:
    """Blend the pipeline scorer with the independent refinement scorer.

    Returns dict with 'unified_score', 'capped' (bool — red-flag cap applied),
    and 'override_discovery' (bool — final score should replace discovery).
    """
    unified = (discovery_score * UNIFIED_DISCOVERY_WEIGHT
               + refinement_score * UNIFIED_REFINEMENT_WEIGHT)
    capped = False
    if refinement_score < discovery_score * UNIFIED_FLAG_RATIO:
        unified = min(unified, refinement_score + UNIFIED_FLAG_CAP_MARGIN)
        capped = True
    return {
        "unified_score": round(unified, 1),
        "capped": capped,
        "override_discovery": unified < discovery_score * UNIFIED_OVERRIDE_RATIO,
    }


# ════════════════════════════════════════════════════════════════════
# THEORY TOURNAMENT (discovery/theory_competition.py)
# ════════════════════════════════════════════════════════════════════
# Dimension weights for the algorithmic scorer (sum = 1.0). The LLM template
# scores are only a diagnostic baseline — algorithmic content-based scoring
# dominates because LLM template scores were identical for every theory.
TOURNAMENT_DIMENSION_WEIGHTS: dict = {
    "explanatory_power": 0.20,
    "predictive_power": 0.20,
    "falsifiability": 0.15,
    "evidence_support": 0.15,
    "novelty": 0.10,
    "simplicity": 0.10,
    "coherence": 0.10,
}

# Final score per dimension = ALGO × algorithmic + LLM × llm template score.
TOURNAMENT_ALGO_WEIGHT = 0.60
TOURNAMENT_LLM_WEIGHT = 0.40

# Round 3 head-to-head blend: overall = SCORE × current + WINRATE × win_rate.
TOURNAMENT_SCORE_WEIGHT = 0.70
TOURNAMENT_WIN_RATE_WEIGHT = 0.30

# Bonuses and caps used while scoring a theory's content.
TOURNAMENT_CAUSAL_BONUS = 0.05   # reward for explicit causal_pathway/counterfactual

# ════════════════════════════════════════════════════════════════════
# RECURSIVE RUN MODE (run_discovery.py --recurse)
# ════════════════════════════════════════════════════════════════════
# Temperature ladder: broad exploration → refinement → convergence.
RECURSE_TEMPERATURES: tuple = (0.7, 0.4, 0.2)

# Dampened loop score = PREV × prev + CURR × current. Prevents a single
# lucky loop from dominating; keeps best-of growing conservatively.
RECURSE_PREV_WEIGHT = 0.70
RECURSE_CURR_WEIGHT = 0.30

# Convergence: stop iterating when consecutive top-3 theory scores
# consistently differ by less than RECURSE_CONVERGENCE_DELTA.
RECURSE_CONVERGENCE_DELTA = 3.0


# ════════════════════════════════════════════════════════════════════
# LLM PROMPT LIMITS (pipeline `_truncated_llm`)
# ════════════════════════════════════════════════════════════════════
# Max characters passed to any LLM before truncation. Providers accept
# far more than the old hard 4,000; literature context is valuable, so we
# keep head-heavy truncation only as a fuse for run-away prompts.
# Override with LUKA_MAX_PROMPT_CHARS env var.
LLM_MAX_PROMPT_CHARS = int(os.getenv("LUKA_MAX_PROMPT_CHARS", "20000"))
LLM_TRUNCATION_HEAD_FRACTION = 0.85  # keep the start (task + instructions)
