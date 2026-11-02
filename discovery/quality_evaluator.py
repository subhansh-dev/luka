"""Quality evaluator for discovery runs."""
import json
from pathlib import Path
HISTORY_FILE = Path(__file__).resolve().parent.parent / "data" / "quality_history.json"

def evaluate_discovery(report):
    """Evaluate quality of a discovery run."""
    phases = report.get("phases", {})
    scores = {}
    lit = phases.get("literature", {})
    scores["literature"] = min(100, 20 + lit.get("papers_found", 0) * 2)
    kg = phases.get("knowledge_graph", {})
    scores["knowledge_graph"] = min(100, 20 + kg.get("entities", 0) + kg.get("relationships", 0) * 0.5)
    gaps = phases.get("gap_detection", {})
    scores["gap_detection"] = min(100, 30 + gaps.get("total_gaps", 0) * 3)
    mech = phases.get("mechanism_generation", {})
    scores["mechanisms"] = min(100, 20 + mech.get("mechanisms_generated", 0) * 10)
    pred = phases.get("prediction_engine", {})
    scores["predictions"] = min(100, 20 + pred.get("total_predictions", 0) * 8)
    theory = phases.get("theory_competition", {})
    scores["theories"] = min(100, 20 + theory.get("theories_compared", 0) * 10)
    weights = {"literature": 0.15, "knowledge_graph": 0.15, "gap_detection": 0.15,
               "mechanisms": 0.20, "predictions": 0.15, "theories": 0.20}
    total = round(sum(scores.get(k, 0) * w for k, w in weights.items()), 1)
    grade = "A" if total >= 80 else "B" if total >= 60 else "C" if total >= 40 else "D" if total >= 20 else "F"
    result = {"total_score": total, "grade": grade, "scores": scores,
              "weak_areas": [k for k, v in scores.items() if v < 30],
              "strong_areas": [k for k, v in scores.items() if v >= 70]}
    try:
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8")) if HISTORY_FILE.exists() else []
        history.append({"topic": report.get("topic", "?"), "score": total, "grade": grade})
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(history[-50:], indent=2), encoding="utf-8")
    except Exception: pass
    return result

def get_improvement_trends():
    try:
        if not HISTORY_FILE.exists(): return {"trend": "no_data"}
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if len(history) < 2: return {"trend": "insufficient_data", "runs": len(history)}
        scores = [h["score"] for h in history]
        recent = sum(scores[-5:]) / len(scores[-5:])
        older = sum(scores[:-5]) / max(1, len(scores[:-5]))
        trend = "improving" if recent > older + 5 else "declining" if recent < older - 5 else "stable"
        return {"trend": trend, "runs": len(history), "recent_avg": round(recent, 1), "best": max(scores)}
    except Exception: return {"trend": "error"}

def is_available(): return True
