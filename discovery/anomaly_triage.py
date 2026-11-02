"""Anomaly triage - assess importance of discovered anomalies."""

IMPORTANCE_CRITERIA = {
    "novelty": {"weight": 0.25},
    "evidence_strength": {"weight": 0.25},
    "impact": {"weight": 0.20},
    "testability": {"weight": 0.15},
    "consistency": {"weight": 0.15},
}

PRIORITY_LEVELS = {
    "CRITICAL": {"min_score": 80, "action": "Immediate human review required"},
    "HIGH": {"min_score": 60, "action": "Investigate further in next run"},
    "MEDIUM": {"min_score": 40, "action": "Log for future reference"},
    "LOW": {"min_score": 20, "action": "Archive"},
    "NOISE": {"min_score": 0, "action": "Discard"},
}

def triage_anomaly(anomaly, context=None):
    scores = {}
    reasoning = []
    name = anomaly.get("name", "Unknown")
    desc = anomaly.get("description", "").lower()
    anomaly_type = anomaly.get("type", "unexplained")

    novelty = {"contradiction": 70, "unexplained": 60, "unexpected": 65, "inconsistency": 55}.get(anomaly_type, 50)
    if "never" in desc or "first" in desc: novelty += 15
    scores["novelty"] = min(100, novelty)

    n_evidence = len(anomaly.get("evidence", []))
    scores["evidence_strength"] = min(80, 20 + n_evidence * 15)

    impact = 40
    if any(w in desc for w in ["biosignature", "habitable", "life", "organic"]): impact += 30
    if any(w in desc for w in ["dark matter", "dark energy", "new physics"]): impact += 25
    if any(w in desc for w in ["exoplanet", "atmosphere", "transit"]): impact += 15
    scores["impact"] = min(100, impact)

    testability = 40
    if any(w in desc for w in ["predict", "observable", "measurable", "detect"]): testability += 30
    if any(w in desc for w in ["jwst", "telescope"]): testability += 20
    scores["testability"] = min(100, testability)

    consistency = 70
    if any(w in desc for w in ["violate", "contradict", "impossible"]): consistency -= 30
    scores["consistency"] = max(0, min(100, consistency))

    total = round(sum(scores[k] * IMPORTANCE_CRITERIA[k]["weight"] for k in scores), 1)
    priority = "NOISE"
    action = "Discard"
    for level, config in sorted(PRIORITY_LEVELS.items(), key=lambda x: x[1]["min_score"], reverse=True):
        if total >= config["min_score"]:
            priority = level
            action = config["action"]
            break

    return {"anomaly_name": name, "priority": priority, "total_score": total,
            "scores": scores, "recommended_action": action}

def triage_batch(anomalies, context=None):
    results = [triage_anomaly(a, context) for a in (anomalies or []) if isinstance(a, dict)]
    results.sort(key=lambda x: x["total_score"], reverse=True)
    return {"total": len(results),
            "critical": sum(1 for r in results if r["priority"] == "CRITICAL"),
            "high": sum(1 for r in results if r["priority"] == "HIGH"),
            "ranked": results}

def is_available(): return True
