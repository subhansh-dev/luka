"""Mission log - structured discovery output."""
import json, time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "mission_logs"

class MissionLog:
    def __init__(self, mission_id=None):
        self.mission_id = mission_id or f"MISSION-{int(time.time())}"
        self.entries = []
        self.discoveries = []
        self.failures = []
        self.pending_review = []
        self.start_time = time.time()

    def log(self, category, message, data=None):
        self.entries.append({"time": datetime.now().isoformat(), "category": category, "message": message})

    def log_discovery(self, name, description, evidence, confidence, priority="MEDIUM"):
        self.discoveries.append({"time": datetime.now().isoformat(), "name": name,
            "description": description[:500], "confidence": confidence, "priority": priority})
        self.log("DISCOVERY", f"{priority}: {name}")

    def log_failure(self, phase, error, impact="unknown"):
        self.failures.append({"time": datetime.now().isoformat(), "phase": phase, "error": str(error)[:200]})
        self.log("FAILURE", f"{phase}: {str(error)[:80]}")

    def log_review_needed(self, item, reason):
        self.pending_review.append({"time": datetime.now().isoformat(), "item": item, "reason": reason})
        self.log("REVIEW", f"Needs review: {item[:80]}")

    def generate_briefing(self):
        elapsed = time.time() - self.start_time
        lines = [f"MISSION BRIEFING: {self.mission_id}", f"Duration: {elapsed:.0f}s", ""]
        if self.discoveries:
            lines.append(f"DISCOVERIES ({len(self.discoveries)}):")
            for d in self.discoveries:
                lines.append(f"  [{d['priority']}] {d['name']} (confidence: {d['confidence']:.0%})")
        else:
            lines.append("DISCOVERIES: None")
        lines.append("")
        if self.failures:
            lines.append(f"FAILURES ({len(self.failures)}):")
            for f in self.failures:
                lines.append(f"  [{f['phase']}] {f['error'][:80]}")
        else:
            lines.append("FAILURES: None")
        lines.append("")
        if self.pending_review:
            lines.append(f"NEEDS HUMAN REVIEW ({len(self.pending_review)}):")
            for r in self.pending_review:
                lines.append(f"  - {r['item'][:80]}: {r['reason'][:80]}")
        return "\n".join(lines)

    def save(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"{self.mission_id}.json"
        data = {"mission_id": self.mission_id, "entries": self.entries,
                "discoveries": self.discoveries, "failures": self.failures,
                "pending_review": self.pending_review, "briefing": self.generate_briefing()}
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return str(path)

def is_available(): return True
