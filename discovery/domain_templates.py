"""
domain_templates.py — Domain-Specific Research Templates for Luka

Different domains have different research patterns, methodologies,
and key metrics. This module provides domain-specific templates that
customize the pipeline's behavior for each scientific field.

Templates define:
- Research question patterns
- Methodology preferences
- Key metrics and validation criteria
- Scoring weight adjustments
- Data source priorities
"""

from typing import Dict, List, Optional


DOMAIN_TEMPLATES = {
    "space_astronomy": {
        "research_patterns": [
            "Is {observation} consistent with {theory}?",
            "Can {mechanism} explain {anomaly} in {dataset}?",
            "What hidden variables drive {correlation} between {var1} and {var2}?",
            "Does {biosignature} indicate {process} on {body}?",
        ],
        "methodology_preferences": [
            "Spectroscopic analysis", "Transit photometry", "Radial velocity",
            "Monte Carlo simulation", "Bayesian inference", "MCMC sampling",
            "Radiative transfer modeling", "Atmospheric retrieval",
        ],
        "key_metrics": [
            "Signal-to-noise ratio (SNR)", "Transit depth (ppm)",
            "Atmospheric mixing ratio", "Equilibrium temperature",
            "Stellar flux", "Orbital period", "Eccentricity",
        ],
        "data_sources": ["NASA Exoplanet Archive", "MAST", "arXiv astro-ph", "SIMBAD", "VizieR", "ESA Sky"],
        "scoring_weights": {
            "novelty": 0.20,      # Astronomy values novel discoveries highly
            "evidence": 0.25,     # Strong observational evidence needed
            "predictions": 0.25,  # Testable predictions critical
            "mathematical": 0.15, # Equations important for modeling
            "mechanism": 0.10,    # Mechanisms can be speculative
            "parsimony": 0.05,    # Simpler explanations preferred
        },
        "validation_criteria": [
            "Reproducible with different instruments",
            "Consistent across multiple wavelengths",
            "Matches known physical laws",
            "Predicts new observable signatures",
        ],
    },

    "isro_earth_observation": {
        "research_patterns": [
            "Can {satellite_data} predict {monsoon_pattern} in {region}?",
            "How does {parameter} from {instrument} correlate with {ground_truth}?",
            "What anomalies in {ocean_data} indicate {phenomenon}?",
            "Does {insat_product} improve forecast accuracy over {baseline} for {region}?",
            "How does {soil_moisture} from {satellite} relate to {crop_yield} in {state}?",
        ],
        "methodology_preferences": [
            "Satellite image analysis", "Time series analysis",
            "Spatial interpolation", "Machine learning classification",
            "Change detection", "NDVI analysis", "Cyclone tracking",
            "Monsoon onset prediction", "Flood inundation mapping",
            "Ocean color analysis", "Scatterometry wind retrieval",
        ],
        "key_metrics": [
            "Sea Surface Temperature (SST) in degC",
            "Rainfall rate (mm/day)",
            "Soil Moisture (m3/m3)",
            "Chlorophyll-a concentration (mg/m3)",
            "Normalized Difference Vegetation Index (NDVI)",
            "Cloud top temperature (K)",
            "Wind speed at 10m (m/s)",
            "Significant wave height (m)",
            "Sea surface salinity (PSU)",
            "Integrated water vapour (mm)",
        ],
        "data_sources": [
            "MOSDAC", "Bhuvan", "VEDAS",
            "arXiv (physics.ao-ph, astro-ph.EP)",
            "ISRO Data Portal", "IMD (India Meteorological Dept)",
        ],
        "scoring_weights": {
            "novelty": 0.15,       # Novel applications of ISRO data
            "evidence": 0.35,      # Observational data is king
            "predictions": 0.20,   # Testable predictions important
            "mathematical": 0.10,  # Models useful but not primary
            "mechanism": 0.10,     # Physical mechanisms matter
            "parsimony": 0.10,     # Simple explanations preferred
        },
        "validation_criteria": [
            "Ground truth validation with IMD station data",
            "Cross-validation between ISRO and international satellites",
            "Consistent across multiple passes/instruments",
            "Matches known Indian monsoon/climate patterns",
            "Predicts verifiable near-term observations",
        ],
    },

    "solar_system": {
        "research_patterns": [
            "What {mechanism} drives {surface_feature} on {body}?",
            "Can {crater} analysis date the {region} of {planet}?",
            "Does {atmosphere} composition suggest {interior_process}?",
        ],
        "methodology_preferences": [
            "Geological mapping", "Impact crater counting",
            "Ephemeris calculation", "Spectral mineralogy",
            "Gravimetric modeling", "Magnetospheric mapping",
        ],
        "key_metrics": [
            "Surface age (Gyr)", "Albedo", "Mineral abundance",
            "Atmospheric pressure (bar)", "Magnetic field strength (nT)",
            "Heliocentric distance (AU)",
        ],
        "data_sources": ["JPL Horizons", "USGS Planetary Data", "NASA PDS", "IAU Gazetteer"],
        "scoring_weights": {
            "novelty": 0.20,
            "evidence": 0.30,
            "predictions": 0.15,
            "mathematical": 0.10,
            "mechanism": 0.15,
            "parsimony": 0.10,
        },
        "validation_criteria": [
            "Physics-consistent orbital parameters",
            "Matches in-situ spacecraft measurements",
            "Geologically plausible mechanisms",
            "Consistent with solar system formation models",
        ],
    },

    "climate_energy": {
        "research_patterns": [
            "How does {solar_activity} impact {parameter} measured by {satellite}?",
            "Can space-derived {variable} improve {model} of {region}?",
        ],
        "methodology_preferences": [
            "Satellite data assimilation", "Trend analysis",
            "Correlation mapping", "Solar radiation modeling",
        ],
        "key_metrics": [
            "Insolation (kWh/m2/day)", "Temperature anomaly (K)",
            "Cloud fraction", "Aerosol optical depth",
        ],
        "data_sources": ["NASA POWER", "NOAA NCEI", "CAMS"],
        "scoring_weights": {
            "novelty": 0.10,
            "evidence": 0.40,
            "predictions": 0.20,
            "mathematical": 0.10,
            "mechanism": 0.10,
            "parsimony": 0.10,
        },
        "validation_criteria": [
            "Correlates with ground-based sensors",
            "Statistically significant trends",
            "Peer-reviewed satellite calibration",
        ],
    },
}


def get_domain_template(domain: str) -> dict:
    """Get the research template for a domain."""
    return DOMAIN_TEMPLATES.get(domain, DOMAIN_TEMPLATES.get("space_astronomy", {}))


def get_scoring_weights(domain: str) -> dict:
    """Get domain-specific scoring weights."""
    template = get_domain_template(domain)
    return template.get("scoring_weights", {
        "novelty": 0.15, "evidence": 0.25, "predictions": 0.20,
        "mathematical": 0.15, "mechanism": 0.15, "parsimony": 0.10,
    })


def get_research_question_prompt(domain: str, topic: str) -> str:
    """Generate domain-specific research question framing."""
    template = get_domain_template(domain)
    patterns = template.get("research_patterns", [])
    metrics = template.get("key_metrics", [])
    methods = template.get("methodology_preferences", [])

    lines = []
    lines.append(f"DOMAIN CONTEXT ({domain}):")
    if patterns:
        lines.append("Research question patterns in this domain:")
        for p in patterns[:3]:
            lines.append(f"  - {p}")
    if metrics:
        lines.append(f"Key metrics: {', '.join(metrics[:5])}")
    if methods:
        lines.append(f"Preferred methods: {', '.join(methods[:4])}")
    return "\n".join(lines)


def get_validation_prompt(domain: str) -> str:
    """Get domain-specific validation criteria for the prompt."""
    template = get_domain_template(domain)
    criteria = template.get("validation_criteria", [])
    if not criteria:
        return ""
    lines = ["Validation criteria for this domain:"]
    for c in criteria:
        lines.append(f"  - {c}")
    return "\n".join(lines)


def adjust_discovery_score(score: dict, domain: str) -> dict:
    """Adjust discovery score using domain-specific weights."""
    weights = get_scoring_weights(domain)
    component_scores = {
        "novelty": score.get("novelty_score", 50),
        "evidence": score.get("evidence_score", 50),
        "predictions": score.get("prediction_score", 50),
        "mathematical": score.get("mathematical_score", 50),
        "mechanism": score.get("mechanism_score", 50),
        "parsimony": score.get("parsimony_score", 50),
    }
    weighted = sum(component_scores[k] * weights.get(k, 0.167) for k in component_scores)
    score["domain_weighted_score"] = round(weighted, 1)
    score["domain_weights"] = weights
    return score
