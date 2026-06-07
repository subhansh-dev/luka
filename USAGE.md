# Luka Usage Guide

This guide provides examples and instructions for running space discovery tasks with Luka.

## 1. Quick Discovery (Phases 1-5)
Use this mode for a fast overview of a topic, literature search, and initial gap detection.

```bash
python run_discovery.py "James Webb observations of TRAPPIST-1" --domain space_astronomy --mode quick
```

## 2. Full Scientific Discovery (Phases 1-16)
The complete pipeline including mechanism generation, theory tournaments, and adversarial skepticism.

```bash
python run_discovery.py "gravitational wave signatures of primordial black holes" --domain space_astronomy --mode full
```

## 3. ISRO Earth Observation
Specific focus on Indian satellite data and meteorological parameters.

```bash
python run_discovery.py "cyclone intensity in the Arabian Sea" --domain isro_earth_observation --mode standard
```

## 4. Solar System Exploration
Geological and orbital analysis of planets, moons, and asteroids.

```bash
python run_discovery.py "subsurface oceans on Europa" --domain solar_system --mode full
```

## 5. Web Dashboard
After any discovery run, Luka generates a report. You can view the cumulative knowledge graph and latest reports via the web dashboard:

1. Locate the `discovery/dashboard/index.html` file.
2. Open it in any modern web browser.
3. Browse the "Theories", "Gaps", and "Knowledge Graph" tabs.

## 6. Self-Improvement & Continuous Mode
To let Luka research autonomously and improve her own internal scoring models:

```bash
# In a Python environment
from discovery.continuous_operation import run_continuous_discovery
run_continuous_discovery(max_runs=5)
```

## 7. Troubleshooting APIs
If a discovery fails during the enrichment phase, check your API keys in `config/api_keys.json`. You can verify reachability with:

```bash
python verify_install.py
```
