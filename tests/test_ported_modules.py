"""
test_ported_modules.py - Tests for modules ported from RUMI to Luka.
"""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def test_computation_engine_constants():
    from discovery.computation_engine import PHYSICS_CONSTANTS, KNOWN_EQUATIONS
    assert "c" in PHYSICS_CONSTANTS
    assert PHYSICS_CONSTANTS["c"]["value"] == 3e8
    assert len(KNOWN_EQUATIONS) >= 10
    print("  [PASS] computation_engine: constants")

def test_computation_engine_verify():
    from discovery.computation_engine import verify_mechanism_numbers
    mechanism = {"name": "test", "description": "B = 1e-6 T", "key_parameters": [{"name": "B", "expected_value": "1e-6 T", "source": "estimated"}], "steps": [], "derivation": ""}
    result = verify_mechanism_numbers(mechanism)
    assert "verified_params" in result
    print("  [PASS] computation_engine: verify")

def test_computation_engine_sanity():
    from discovery.computation_engine import _sanity_check_estimate
    r = _sanity_check_estimate("velocity", "5e8", "v = 5e8 m/s")
    assert r["status"] == "checked"
    r2 = _sanity_check_estimate("velocity", "1e5", "v = 1e5 m/s")
    assert r2["status"] == "reasonable"
    print("  [PASS] computation_engine: sanity")

def test_computation_engine_parse():
    from discovery.computation_engine import _parse_number
    assert _parse_number("3e8") == 3e8
    assert _parse_number("42") == 42.0
    assert _parse_number("abc") is None
    print("  [PASS] computation_engine: parse")

def test_math_engine_extract():
    from discovery.math_engine import _extract_equations
    text = "The flux F = 1.5e-6 Jy was measured. Mass M = 1.4 M_sun."
    eqs = _extract_equations(text)
    assert len(eqs) >= 1
    print("  [PASS] math_engine: extract")

def test_math_engine_reasonableness():
    from discovery.math_engine import _check_physical_reasonableness
    assert "reasonable" in _check_physical_reasonableness(1.989e30, "kg", "mass")
    assert "ABOVE" in _check_physical_reasonableness(5e8, "m/s", "speed")
    print("  [PASS] math_engine: reasonableness")

def test_math_engine_dimensional():
    from discovery.math_engine import check_dimensional_consistency
    r = check_dimensional_consistency("F = 10 N. E = 5 J.")
    assert "equations_checked" in r
    print("  [PASS] math_engine: dimensional")

def test_math_engine_monte_carlo():
    from discovery.math_engine import monte_carlo_validate
    r = monte_carlo_validate({"mean": 0.5, "std": 0.1, "expected": 0.5, "tolerance": 0.2})
    assert r["status"] == "validated"
    print("  [PASS] math_engine: monte carlo")

def test_math_engine_full():
    from discovery.math_engine import run_math_verification
    r = run_math_verification({"description": "L = 3.828e26 W", "mathematical_model": "L = 4 * pi * R**2 * sigma_SB * T**4"})
    assert r["overall_score"] >= 10
    print("  [PASS] math_engine: full pipeline")

def test_mechanism_completeness():
    from discovery.mechanism_completeness import MechanismCompletenessChecker
    c = MechanismCompletenessChecker()
    m = [{"name": "test", "description": "Tidal heating", "mathematical_model": "P = G*M**2*R**5", "steps": ["Step 1: X", "Therefore Y"], "key_parameters": [{"name": "e", "value": "0.01"}], "derivation": "Starting from first principles"}]
    r = c.check_mechanisms(m, domain="physics")
    assert r["total"] == 1
    assert r["avg_completeness"] > 0.3
    print("  [PASS] mechanism_completeness")

def test_observability_checker():
    from discovery.observability_checker import ObservabilityChecker, INSTRUMENT_LIMITS
    c = ObservabilityChecker()
    r = c.check_predictions([{"statement": "strain h ~ 1e-22 detectable by LIGO"}], domain="physics")
    assert r["total"] == 1
    assert "gravitational_wave" in INSTRUMENT_LIMITS
    print("  [PASS] observability_checker")

def test_curious_questioning():
    from discovery.curious_questioning import CuriousQuestioning
    cq = CuriousQuestioning()
    papers = [{"title": "Surprising phosphine detection on Venus",
               "abstract": "We report the surprising detection of phosphine in Venus atmosphere. This contradicts current atmospheric models. However the signal remains unclear and not well understood."}]
    obs = cq._extract_observations("phosphine detection Venus", papers)
    assert len(obs) >= 1, f"Expected 1+ observations, got {len(obs)}"
    qs = cq._generate_questions("phosphine detection Venus", "space_astronomy", obs)
    cats = set(q.get("category") for q in qs if isinstance(q, dict))
    assert len(cats) >= 2
    result = cq.run("exoplanet atmospheric biosignatures", "space_astronomy")
    assert "questions" in result
    print("  [PASS] curious_questioning")

def test_math_chain():
    from discovery.math_chain import MathChain
    mc = MathChain()
    r = mc.process_mechanisms([{"name": "test", "description": "tidal heating"}], "Europa", "solar_system")
    assert len(r) == 1
    print("  [PASS] math_chain")

def test_prediction_validator():
    from discovery.prediction_literature_validator import PredictionLiteratureValidator
    v = PredictionLiteratureValidator()
    claims = v._extract_numerical_claims("P ~ 0.12 with efficiency ~ 85%")
    assert len(claims) >= 1
    r = v.validate_predictions([])
    assert r["summary"]["total_predictions"] == 0
    adj = v._compute_confidence_adjustment({"validation_status": "supported", "supporting_papers": 3}, 0.5, 5)
    assert adj == 0.10
    print("  [PASS] prediction_literature_validator")

if __name__ == "__main__":
    tests = [test_computation_engine_constants, test_computation_engine_verify, test_computation_engine_sanity, test_computation_engine_parse, test_math_engine_extract, test_math_engine_reasonableness, test_math_engine_dimensional, test_math_engine_monte_carlo, test_math_engine_full, test_mechanism_completeness, test_observability_checker, test_curious_questioning, test_math_chain, test_prediction_validator]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
    print()
    print("=" * 50)
    print("RESULTS: %d passed, %d failed out of %d" % (passed, failed, len(tests)))
    print("=" * 50)
