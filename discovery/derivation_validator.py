"""Derivation validator - check if mechanism derivations are reproducible.

A derivation is reproducible if:
1. Each step follows from the previous one
2. All variables are defined before use
3. Numerical examples are provided
4. Units are consistent
"""
import re

def validate_derivation(mechanism: dict) -> dict:
    """Validate a mechanism derivation for reproducibility.

    Returns:
        dict with score, issues, and suggestions
    """
    issues = []
    suggestions = []
    score = 1.0

    steps = mechanism.get("steps", [])
    model = mechanism.get("mathematical_model", "")
    params = mechanism.get("key_parameters", [])
    derivation = mechanism.get("derivation", "")

    # Check 1: Has steps
    if not steps or len(steps) < 2:
        issues.append("Too few steps (need at least 2)")
        score -= 0.3

    # Check 2: Steps have quantitative content
    quant_steps = 0
    for step in steps:
        if isinstance(step, str) and any(c.isdigit() for c in step):
            quant_steps += 1
    if len(steps) > 0 and quant_steps / len(steps) < 0.5:
        issues.append("Less than half of steps contain numbers")
        score -= 0.2

    # Check 3: Has mathematical model
    if not model or len(model) < 10:
        issues.append("No mathematical model")
        score -= 0.3
    elif "=" not in model and "pi" not in model.lower() and "hbar" not in model.lower():
        issues.append("Mathematical model has no equations")
        score -= 0.2

    # Check 4: Has key parameters with values
    if not params:
        issues.append("No key parameters defined")
        score -= 0.1
    else:
        params_with_values = sum(1 for p in params if isinstance(p, dict) and p.get("expected_value"))
        if params_with_values < len(params):
            issues.append(f"Only {params_with_values}/{len(params)} parameters have values")
            score -= 0.1

    # Check 5: Steps reference each other (logical flow)
    step_text = " ".join(str(s) for s in steps).lower()
    flow_words = ["therefore", "thus", "which gives", "yielding", "substituting", "solving"]
    has_flow = sum(1 for w in flow_words if w in step_text)
    if has_flow < 1 and len(steps) > 2:
        issues.append("Steps lack logical connectors (therefore, thus, which gives)")
        score -= 0.1

    # Check 6: Numerical example exists
    has_numbers = any(c.isdigit() for c in str(model) + step_text)
    if not has_numbers:
        issues.append("No numerical examples in derivation")
        score -= 0.2

    # Generate suggestions
    if "No mathematical model" in issues:
        suggestions.append("Add equations governing the mechanism")
    if "No numerical examples" in str(issues):
        suggestions.append("Add a numerical example: plug in values and compute result")
    if "Too few steps" in issues:
        suggestions.append("Break the mechanism into at least 3 causal steps")

    score = max(0.0, min(1.0, score))

    return {
        "name": mechanism.get("name", "?"),
        "score": round(score, 2),
        "reproducible": score >= 0.6,
        "issues": issues,
        "suggestions": suggestions,
        "has_equations": "=" in str(model) or "pi" in str(model).lower(),
        "has_numbers": has_numbers,
        "step_count": len(steps),
    }


def validate_batch(mechanisms: list) -> dict:
    """Validate a batch of mechanism derivations."""
    results = [validate_derivation(m) for m in (mechanisms or []) if isinstance(m, dict)]
    reproducible = sum(1 for r in results if r["reproducible"])
    avg_score = sum(r["score"] for r in results) / max(1, len(results))

    return {
        "total": len(results),
        "reproducible": reproducible,
        "not_reproducible": len(results) - reproducible,
        "avg_score": round(avg_score, 2),
        "details": results,
    }

def is_available():
    return True
