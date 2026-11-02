# Luka

<p align="center">
  <a href="https://github.com/subhansh-dev/luka"><img src="https://img.shields.io/github/license/subhansh-dev/luka?color=green" alt="MIT License" /></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.13%2B-blue" alt="Python 3.13+" /></a>
  <a href="https://github.com/subhansh-dev/luka/stargazers"><img src="https://img.shields.io/github/stars/subhansh-dev/luka?style=flat" alt="Stars" /></a>
  <a href="https://github.com/subhansh-dev/luka/issues"><img src="https://img.shields.io/github/issues/subhansh-dev/luka" alt="Issues" /></a>
  <a href="https://github.com/subhansh-dev/luka/pulls"><img src="https://img.shields.io/github/issues-pr/subhansh-dev/luka" alt="Pull requests" /></a>
</p>

Luka is an autonomous scientific discovery engine that runs in your terminal. You give it a research topic, it reads the actual literature (arXiv, Semantic Scholar, OpenAlex, PubMed, Crossref), builds a knowledge graph, hunts for gaps and anomalies in what's known, proposes hidden variables and causal mechanisms, runs a tournament between competing theories, attacks every surviving theory adversarially, and verifies what's left with real math (sympy, numerical sweeps, uncertainty propagation).

It started as a fork of [RUMI](https://github.com/subhansh-dev/rumi) specialized for space science and grew into its own thing: a 38-phase pipeline with a declarative mode system, persistent memory across runs, and scoring that's now centralized in one auditable file.

One honest note up front: Luka is LLM-driven, so its output is *hypotheses with receipts*, not proven science. Every claim it makes is supposed to carry the papers and computations behind it, but you should read those yourself. That's the point.

## The pipeline

The pipeline is a phase registry. Each phase is a standalone function over a shared context, and each declares the minimum mode that includes it: `quick` runs 6 foundation phases, `standard` adds analysis (10 total), `full` runs all 38, including the expensive verification work.

| Stage group | What actually happens |
|---|---|
| Literature | Multi-round search across 4 sources, then a 2-hop citation walk to pull in papers the keyword search missed |
| Knowledge graph | Entity + relationship extraction, live enrichment from domain APIs (NASA, SIMBAD, VizieR, JPL Horizons, ESA Sky and friends), graph inference to weld fragmented nodes together |
| Gap + anomaly detection | What's missing from the graph, what doesn't fit existing explanations |
| Hidden variables | Proposing unseen entities or processes that would explain the anomalies, with parameters and their origins |
| Mechanisms | Causal pathways with equations derived and checked (derivation validator, math chain) |
| Predictions | Testable predictions, filtered by observability and against published data (contradicted ones get dropped) |
| Theory tournament | 20ish candidates compete, weak ones get eliminated across rounds, survivors get head-to-head cross-comparison |
| Adversarial gate | Every theory gets attacked. Theories that die are actually removed from the pool now, and if the tournament winner gets killed, the best survivor is promoted |
| Verification | Sympy equation verification, dimensional analysis, parameter sweeps, consistency checks. Casimir/gold reference computations only run on topics that actually involve them |
| Skeptic + scoring | Fair-critique review, discovery scoring, cross-validation against held-out papers, a 13-stage refinement pass with an independent "courtroom" verdict that can pull the final grade down |
| Self-improvement | Reflexion on what went wrong, and the pipeline proposes its own next research questions |

All the scoring weights, blends, and grade thresholds live in [`discovery/scoring_config.py`](discovery/scoring_config.py). If you want to argue with how Luka judges theories, argue with that file.

## Domains

Five domains ship with their own entity types, enrichment APIs, search templates, and datasets:

| Domain | Examples of what runs |
|---|---|
| `space_astronomy` | Exoplanets, stellar spectra; NASA Exoplanet Archive data analysis |
| `isro_earth_observation` | Monsoon, satellite imagery, Indian missions (INSAT, Oceansat) |
| `solar_system` | Craters, asteroid deflection; JPL Horizons, MPC, USGS planetary data |
| `climate_energy` | Solar radiation (NASA POWER), satellite-derived climate data |
| `general` | Anything else, no domain tuning |

Domain is auto-detected from the topic, or you pass `--domain`. Data analysis is config-driven: a domain lists its `data_sources` in `discovery/domains.py` and the pipeline fetches and analyzes them.

## Memory across runs

Luka keeps two memories: a discovery archive of past runs (loaded at startup so it doesn't repeat itself) and a hypothesis memory in SQLite, where theories from every run land with confidence scores. Next run on a related topic, prior hypotheses get fed back in as "build on these, refine them, or find what they missed."

## Getting started

```bash
git clone https://github.com/subhansh-dev/luka
cd luka
pip install -r requirements.txt
python verify_install.py
```

You'll want Python 3.13+. API keys go in `config/api_keys.json` (not committed, gitignored):

```json
{
  "openrouter_api_key": "sk-or-...",
  "nvidia_api_key": "nvapi-...",
  "cerebras_api_key": "...",
  "gemini_api_key": "..."
}
```

All of these are optional; the pipeline runs best with at least one of OpenRouter or NVIDIA, and falls back across providers mid-run if one hits a rate limit. NASA endpoints work with `DEMO_KEY`. `verify_install.py` will tell you what's reachable.

## Running it

On Windows, the easy way:

```bat
run.bat
```

That opens the terminal UI (Rich + prompt_toolkit). Type `/discover "your topic"` inside it, `/dashboard` to watch live results. `run.bat some topic` runs the pipeline directly.

Or drive it from the CLI:

```bash
# full 38-phase run
python run_discovery.py "exoplanet atmospheric biosignatures" --domain space_astronomy --mode full

# standard mode: 10 phases, cheaper
python run_discovery.py "monsoon prediction Bay of Bengal" --domain isro_earth_observation --mode standard

# quick sanity check, 6 phases
python run_discovery.py "near-Earth asteroid deflection" --mode quick

# recursive mode: 3 passes at descending temperature, keeps the best result
python run_discovery.py "why do gas giants radiate more heat than they receive" --recurse
```

Reports land in `discovery/sessions/` as JSON, readable without the fancy UI.

There's also a fully autonomous loop that picks its own topics from the research frontier and runs forever:

```bash
python -c "from discovery.continuous_operation import start_loop; start_loop()"
```

## Project layout

```
discovery/          the engine: pipeline, tournament, verifiers, scorers, domains (113 modules)
brain/              cognitive plumbing: memory, planners, reflexion
agents/scientist/   role prompts (literature reviewer, novelty analyst, peer reviewer, ...)
config/             provider config
tests/              pytest suite (run `python -m pytest tests/`)
run.bat             Windows launcher for the terminal UI
```

## Contributing

Issues and PRs welcome. If you're adding a pipeline phase: write the phase function, add it to `PHASE_REGISTRY` in `discovery/discovery_pipeline_v2.py` with a `min_mode`, done. Tests that don't need API keys run locally with `python -m pytest tests/test_pipeline_refactor.py`.

## License

[MIT](LICENSE)
