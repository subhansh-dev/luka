# CHANGES.md — Modules Ported from RUMI to Luka

## Summary

7 modules ported from RUMI discovery pipeline to Luka, adding real mathematical verification, mechanism completeness checking, observability validation, and curious questioning capabilities.

---

## Ported Modules

### 1. discovery/computation_engine.py (516 lines)
**What it does:** Verifies mechanism numbers computationally using a physics constants library, SymPy symbolic verification, derivation tracing, and sanity checks against physical bounds.

**Key features:**
- 20+ physics constants (c, G, h, k_B, etc.) with values and units
- 12 known equations (plasma frequency, Stefan-Boltzmann, Schwarzschild radius, etc.)
- Number extraction from scientific notation (3x10^8, 1.5e-6, etc.)
- Derivation chain tracing through mechanism text
- Sanity checks: speed > c is flagged, T < 0 K is flagged
- SymPy symbolic verification of mathematical models

**How it improves Luka:** Catches unphysical values and verifies that stated numbers actually follow from the derivation.

---

### 2. discovery/math_engine.py (994 lines)
**What it does:** 6-category mathematical verification of theories and mechanisms.

**Key features:**
- Equation solving: Extracts equations from text, evaluates with physical constants
- Dimensional analysis: Checks if equations are dimensionally consistent
- Kinetic equation validation: Detects unbounded factors, singularities, negative concentrations
- Numerical simulation: Solves ODEs with SciPy
- Monte Carlo validation: Statistical verification of numerical claims
- Cross-equation consistency: Uses SymPy to check if equations are mutually consistent
- Physical bounds checking (mass: 1e-35 to 1e55 kg, speed: 0 to 3e8 m/s, etc.)

**How it improves Luka:** Provides rigorous mathematical verification beyond LLM-based scoring.

---

### 3. discovery/math_chain.py (135 lines)
**What it does:** Explicit math derivation chain: Mechanism -> Equation -> Parameter -> Derivation -> Numerical Example.

**Key features:**
- Derives governing equations from mechanism descriptions
- Estimates parameters with provenance (known constant, derived, estimated)
- Numerical validation: plugs in values and checks physical reasonableness

**How it improves Luka:** Forces mechanisms to have explicit equations and parameters instead of qualitative hand-waving.

---

### 4. discovery/mechanism_completeness.py (175 lines)
**What it does:** Checks if mechanisms are derivationally complete.

**Key features:**
- Domain-aware scoring: physics checks equations/derivations, biology checks pathways/causal chains
- 6 completeness checks: assumptions, derivation, transport coefficients, numerical validation, parameters, simulation
- Gap identification: lists what is missing from each mechanism

**How it improves Luka:** Distinguishes rigorous mechanisms from hand-waving.

---

### 5. discovery/observability_checker.py (63 lines)
**What it does:** Checks if predictions are actually measurable with real instruments.

**Key features:**
- Instrument limits for: HST/Euclid, VLT/Keck, Chandra/XMM, LIGO/Virgo, PPMS, SQUID, dilution fridge, diamond anvil
- Compares predicted signal magnitudes against instrument floors
- Returns accessibility rate

**How it improves Luka:** A prediction you cannot test is not a prediction.

---

### 6. discovery/prediction_literature_validator.py (405 lines)
**What it does:** Validates numerical predictions against real published literature.

**Key features:**
- Extracts specific numerical claims from predictions
- Builds targeted search queries from claims and mechanism parameters
- Searches literature for papers that TESTED those exact claims
- LLM comparison of numbers against published data
- Confidence adjustment: +0.10 for supported, -0.20 for contradicted

**How it improves Luka:** Grounds predictions in real data.

---

### 7. discovery/curious_questioning.py (497 lines)
**What it does:** Phase 0 Newton Step — generates deep questions that drive discovery before literature review.

**Key features:**
- 5 question categories: Assumption Attack, Missing Variable, Scale Transfer, Cross-Domain, Anomaly
- Observation extraction from papers
- Cross-domain generalization: Newton move — does the same phenomenon appear elsewhere?
- Constraint generation for the pipeline

**How it improves Luka:** Starts with targeted questions instead of broad search.

---

## Pipeline Integration

| Sub-phase | Module | When it runs |
|-----------|--------|--------------|
| Phase 0 | curious_questioning | Before literature search |
| Phase 6.5 | math_chain | After mechanism generation |
| Phase 6.6 | mechanism_completeness | After math chain |
| Phase 7.5 | observability_checker | After prediction engine |
| Phase 7.6 | prediction_literature_validator | After observability check |
| Phase 9.3 | math_engine | After computational verification |
| Phase 9.4 | computation_engine | After enhanced math verification |

---

## Intentionally Excluded

### API Wrappers (not ported)
alphafold.py, clinicaltrials.py, drugbank_mini.py, kegg.py, ligo.py, molecule.py, nist_api.py, opencitations.py, openfda.py, pdb.py, pubchem.py, uniprot.py, inspire_hep.py

**Reason:** Domain-specific API wrappers for drug discovery and physics databases. Luka is specialized for space discovery. These add no algorithmic value.

### Thin LLM Wrappers (not ported)
cross_validation.py, empc_pipeline.py

**Reason:** Just wrap LLM calls with fancy names. Luka already has equivalent modules.

### Recurrent Loops (not ported)
Phase 6-8 looping 3 times

**Reason:** Luka handles iterative refinement via the --iterate flag in run_discovery.py.

---

## Test Results

14 passed, 0 failed out of 14 tests

- computation_engine: constants, verify, sanity, parse
- math_engine: extract, reasonableness, dimensional, monte carlo, full pipeline
- mechanism_completeness: physics and biology domains
- observability_checker: basic checking and instrument limits
- curious_questioning: extraction, categories, full pipeline
- math_chain: passthrough without LLM
- prediction_literature_validator: extraction, empty result, confidence adjustment
