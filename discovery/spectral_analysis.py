"""Spectral analysis for identifying molecules from astronomical spectra."""
import math

# Known molecular absorption lines (wavelengths in micrometers)
MOLECULAR_LINES = {
    "H2O": {"lines": [1.4, 1.9, 2.7, 6.3], "type": "molecule", "significance": "water vapor"},
    "CO2": {"lines": [2.0, 2.7, 4.3, 15.0], "type": "molecule", "significance": "carbon dioxide"},
    "CH4": {"lines": [1.7, 2.3, 3.3, 7.7], "type": "molecule", "significance": "methane"},
    "CO": {"lines": [2.3, 4.7], "type": "molecule", "significance": "carbon monoxide"},
    "O3": {"lines": [9.6, 14.2], "type": "molecule", "significance": "ozone"},
    "NH3": {"lines": [1.5, 2.3, 6.1, 10.5], "type": "molecule", "significance": "ammonia"},
    "O2": {"lines": [0.76, 1.27], "type": "molecule", "significance": "oxygen"},
    "N2O": {"lines": [4.5, 7.8, 17.0], "type": "molecule", "significance": "nitrous oxide"},
    "SO2": {"lines": [4.0, 7.3, 8.7, 19.3], "type": "molecule", "significance": "sulfur dioxide"},
    "PH3": {"lines": [4.3, 10.0], "type": "molecule", "significance": "phosphine (biosignature candidate)"},
    "HCN": {"lines": [3.0, 14.0], "type": "molecule", "significance": "hydrogen cyanide"},
    "C2H6": {"lines": [3.3, 12.2], "type": "molecule", "significance": "ethane"},
    "TiO": {"lines": [0.5, 0.7, 0.8], "type": "molecule", "significance": "titanium oxide (hot atmospheres)"},
    "VO": {"lines": [0.5, 0.7, 1.0], "type": "molecule", "significance": "vanadium oxide (hot atmospheres)"},
    "Na": {"lines": [0.589], "type": "element", "significance": "sodium (atmospheric)"},
    "K": {"lines": [0.767], "type": "element", "significance": "potassium (atmospheric)"},
    "He": {"lines": [1.083], "type": "element", "significance": "helium"},
    "Halpha": {"lines": [0.656], "type": "line", "significance": "hydrogen alpha (star formation)"},
    "Hbeta": {"lines": [0.486], "type": "line", "significance": "hydrogen beta"},
    "Lyman_alpha": {"lines": [0.122], "type": "line", "significance": "Lyman alpha (high-z)"},
}

BIOSIGNATURE_MOLECULES = ["O2", "O3", "H2O", "CH4", "PH3", "N2O", "NH3"]

def identify_molecules(wavelengths_um, tolerance=0.05):
    """Identify molecules from absorption line wavelengths.

    Args:
        wavelengths_um: List of observed absorption line wavelengths in micrometers
        tolerance: Matching tolerance in micrometers (default 0.05)

    Returns:
        List of identified molecules with confidence scores
    """
    identifications = []
    for molecule, info in MOLECULAR_LINES.items():
        matched_lines = []
        for obs_wl in wavelengths_um:
            for ref_wl in info["lines"]:
                if abs(obs_wl - ref_wl) <= tolerance:
                    matched_lines.append({"observed": obs_wl, "reference": ref_wl, "deviation": abs(obs_wl - ref_wl)})
                    break
        if matched_lines:
            confidence = min(1.0, len(matched_lines) / max(1, len(info["lines"]) * 0.5))
            is_biosignature = molecule in BIOSIGNATURE_MOLECULES
            identifications.append({
                "molecule": molecule,
                "matched_lines": len(matched_lines),
                "total_lines": len(info["lines"]),
                "confidence": round(confidence, 2),
                "significance": info["significance"],
                "type": info["type"],
                "is_biosignature": is_biosignature,
                "details": matched_lines,
            })
    identifications.sort(key=lambda x: x["confidence"], reverse=True)
    return identifications

def assess_biosignature(identifications):
    """Assess biosignature potential from identified molecules."""
    found_bio = [m for m in identifications if m.get("is_biosignature")]
    if not found_bio:
        return {"score": 0, "assessment": "No biosignature molecules detected"}
    names = [m["molecule"] for m in found_bio]
    score = 0
    notes = []
    if "O2" in names and "CH4" in names:
        score += 40
        notes.append("O2 + CH4 co-presence: strong disequilibrium biosignature")
    if "O3" in names:
        score += 20
        notes.append("O3 detected: proxy for O2")
    if "H2O" in names:
        score += 15
        notes.append("H2O detected: habitability indicator")
    if "PH3" in names:
        score += 25
        notes.append("PH3 detected: potential biosignature (no known abiotic source on rocky planets)")
    if "N2O" in names:
        score += 15
        notes.append("N2O detected: biological origin likely")
    return {
        "score": min(100, score),
        "biosignature_molecules": names,
        "assessment": "; ".join(notes) if notes else "Biosignature molecules present but no strong co-detection",
    }

def is_available():
    return True
