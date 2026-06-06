# Luka — Autonomous Space Discovery Engine

<p align="center">
  <img src="assets/rumi.png" alt="Luka Logo" width="400" />
</p>

<p align="center">
  <a href="https://github.com/subhansh-dev/luka">
    <img src="https://img.shields.io/badge/Luka-Space%20Discovery-blueviolet" alt="Luka" />
  </a>
  <a href="https://python.org/versions/3.13">
    <img src="https://img.shields.io/badge/Python-3.13+-blue" alt="Python" />
  </a>
</p>

<p align="center">
  <b>The World's Best Autonomous Space Discovery Engine</b><br>
  16-Phase Pipeline · ISRO & NASA Integration · Theory Tournament · Adversarial Space Physics · Self-Improving Architecture
</p>

---

## What is Luka?

**Luka** is an autonomous space discovery engine, forked from RUMI and specialized exclusively for the cosmos. Give Luka a research topic in astronomy, exoplanetology, or earth observation, and she performs a deep-space intelligence sweep: reading papers from arXiv astro-ph, fetching data from NASA and ISRO (MOSDAC/Bhuvan), querying astronomical catalogs (SIMBAD/VizieR), and analyzing orbital dynamics via JPL Horizons.

Luka doesn't just summarize; she **discovers**. She builds knowledge graphs of celestial bodies, finds anomalies in light curves, proposes novel astrophysical mechanisms with formal equations, and runs a tournament of competing theories to explain the unknown.

### Key Space Domains

- **Space & Astronomy**: Exoplanets, stars, galaxies, stellar evolution, and deep-space missions.
- **ISRO Earth Observation**: MOSDAC meteorological data, Bhuvan geospatial analysis, and Indian satellite missions (INSAT, Oceansat, Scatsat).
- **Solar System & Planetary Geology**: Planetary surfaces, craters, atmospheres, and asteroid deflection strategies (USGS & IAU data).
- **Climate & Energy**: Space-relevant climate data and satellite-derived solar radiation (NASA POWER).

---

## Discovery Pipeline v2 (Space Edition)

Luka's 16-phase pipeline is tuned for the rigors of astrophysics:

1.  **Literature Search**: arXiv (astro-ph) + Semantic Scholar + OpenAlex.
2.  **Space Enrichment**: Real-time hooks into NASA API, ISRO MOSDAC, SIMBAD, and ESA Sky.
3.  **Knowledge Graph**: Extraction of exoplanets, spectral features, stellar properties, and orbital parameters.
4.  **Anomaly Detection**: Identifying transits that don't fit models or spectral lines that shouldn't be there.
5.  **Hidden Variables**: Proposing unseen perturbers (e.g., Planet Nine, dark matter interactions).
6.  **Mechanisms**: Causal drivers (tidal heating, radiative transfer) with physical equations.
7.  **Predictions**: Testable observational signatures for James Webb (JWST) or Gaia.
8.  **Theory Tournament**: competing models (e.g., Hot Jupiter vs. Brown Dwarf) ranked by evidence.
9.  **Adversarial Review**: A skeptic agent that checks if hypotheses violate the laws of thermodynamics or orbital mechanics.
10. **Data Analysis**: Direct fetching of NASA Exoplanet Archive and ISRO satellite products.
11. **Self-Improvement**: Luka evaluates her own discovery quality and tunes her scoring weights for the next run.

---

## Self-Improving Architecture

Luka is capable of modifying herself to become a better scientist:

- **Self-Assessment Loop**: After each run, Luka evaluates the novelty and strength of her evidence.
- **Code Self-Modification**: Luka can tune her own discovery pipeline parameters and scoring weights based on past success.
- **Knowledge Gap Auto-Discovery**: Luka identifies what questions remain unanswered and prioritizes them for future autonomous runs.

---

## Installation & Verification

### Prerequisites
- Python 3.13+
- NASA API Key (use `DEMO_KEY` by default)
- Internet connection for API reachability

### Setup
```bash
git clone https://github.com/subhansh-dev/luka
cd luka
pip install -r requirements.txt
playwright install chromium
```

### Verify Installation
Run the built-in verifier to check API reachability:
```bash
python verify_install.py
```

---

## Usage

### Run a Discovery
```bash
# Exoplanet research
python run_discovery.py "exoplanet atmospheric biosignatures" --domain space_astronomy --mode full

# ISRO Earth Observation
python run_discovery.py "monsoon prediction Bay of Bengal" --domain isro_earth_observation --mode quick

# Solar System
python run_discovery.py "near-Earth asteroid deflection" --domain solar_system --mode standard
```

### Continuous Orchestration
Luka can run autonomously, picking topics from the research frontier:
```bash
python -c "from discovery.continuous_operation import start_loop; start_loop()"
```

---

## License

[MIT](LICENSE) — Luka is an open-source project dedicated to the advancement of space discovery.
