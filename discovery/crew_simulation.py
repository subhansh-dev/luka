"""Crew simulation - multi-agent debate on findings."""

CREW_ROLES = {
    "navigator": "Where to look next? Prioritize observations.",
    "scientist": "What does the evidence mean? Is it novel?",
    "engineer": "Can we test this? What instruments needed?",
    "skeptic": "What could be wrong? Alternative explanations?",
    "analyst": "What do the numbers say? Statistical significance?",
}

def debate_finding(finding, llm_call=None):
    """Run a crew debate on a finding."""
    if not llm_call:
        return _quick_debate(finding)
    name = finding.get("name", "Unknown")
    desc = finding.get("description", "")[:500]
    prompt = f"""Simulate 5 AI specialists discussing: {name}
Description: {desc}
Evidence items: {len(finding.get("evidence", []))}

Each give 1-2 sentence assessment:
NAVIGATOR: ...
SCIENTIST: ...
ENGINEER: ...
SKEPTIC: ...
ANALYST: ...
CONSENSUS: 1 sentence summary
CONFIDENCE: 0.0-1.0"""
    result = llm_call(prompt, max_tokens=1024)
    if not result: return _quick_debate(finding)
    assessments = {}
    for role in CREW_ROLES:
        for line in result.split("\n"):
            if line.upper().startswith(role.upper() + ":"):
                assessments[role] = line.split(":", 1)[1].strip()
    consensus = ""
    for line in result.split("\n"):
        if line.upper().startswith("CONSENSUS:"):
            consensus = line.split(":", 1)[1].strip()
    confidence = 0.5
    for line in result.split("\n"):
        if line.upper().startswith("CONFIDENCE:"):
            try: confidence = float(line.split(":", 1)[1].strip())
            except: pass
    return {"finding": name, "assessments": assessments, "consensus": consensus,
            "confidence": confidence, "crew_size": len(assessments)}

def _quick_debate(finding):
    name = finding.get("name", "Unknown")
    n_ev = len(finding.get("evidence", []))
    n_pred = len(finding.get("predictions", []))
    confidence = min(0.8, 0.2 + n_ev * 0.1 + n_pred * 0.05)
    return {
        "finding": name,
        "assessments": {
            "navigator": f"Search for confirming observations for {name}",
            "scientist": f"{n_ev} evidence items. {"Promising" if n_ev >= 3 else "Needs more evidence"}.",
            "engineer": f"{"Can be tested" if n_pred > 0 else "Need testable predictions"}",
            "skeptic": f"Check alternatives. {n_ev} items is {"sufficient" if n_ev >= 5 else "insufficient"}.",
            "analyst": f"Need quantitative analysis. {n_pred} predictions available.",
        },
        "consensus": f"{"Promising" if confidence > 0.5 else "Preliminary"} finding needing validation",
        "confidence": round(confidence, 2),
        "crew_size": 5,
    }

def is_available(): return True
