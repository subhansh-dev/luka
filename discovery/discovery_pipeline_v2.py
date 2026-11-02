"""
discovery_pipeline_v2.py — Luka's Genuine Scientific Discovery Pipeline.

The difference from run_full_pipeline.py:
  Old: Papers → Entity Extraction → Knowledge Graph → Hypothesis → Skeptic
  New: Papers → Knowledge Graph → GAPS → ANOMALIES → HIDDEN VARIABLES →
       MECHANISMS → PREDICTIONS → THEORY COMPETITION → COMPUTATIONAL
       VERIFICATION → DISCOVERY SCORING → Discovery Report

This is not a research assistant. This is a discovery engine.

Pipeline:
  Phase 1: LITERATURE — Fetch real papers (arXiv + PubMed)
  Phase 2: KNOWLEDGE GRAPH — Entity extraction + relationship building
  Phase 3: GAP DETECTION — What's missing in the knowledge?
  Phase 4: ANOMALY DETECTION — What doesn't fit existing theories?
  Phase 5: MISSING VARIABLES — Propose hidden entities/processes
  Phase 6: MECHANISM GENERATION — Build causal pathways
  Phase 7: PREDICTION ENGINE — Generate testable predictions
  Phase 8: THEORY COMPETITION — Compare multiple explanations
  Phase 9: COMPUTATIONAL VERIFICATION — Run actual computations
  Phase 10: SKEPTIC REVIEW — Adversarial critique
  Phase 11: DISCOVERY SCORING — Final quality gate
  Phase 12: DISCOVERY REPORT — Structured output

Output format:
  Observation: [what we see]
  Problem: [what existing theories can't explain]
  Gap: [what's missing]
  Proposed Mechanism: [our explanation]
  Predictions: [what should be true if we're right]
  Computational Test: [what we computed]
  Result: [supports/contradicts]
  Discovery Score: [0-100]
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys
import io

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Existing modules
from discovery.llm_client import call as llm_call, call_json, get_status
from discovery.citation_grounding import fetch_papers, build_citation_context, ground_claims
from discovery.computational import run_all_calculations, format_calculations_for_prompt
from discovery.domain_computational import run_domain_calculations, format_for_prompt, DOMAIN_COMPUTATIONS
from discovery.claim_labeler import label_report, generate_confidence_summary
from discovery.claim_provenance import ProvenanceTracker
from discovery.graph import KnowledgeGraph
from discovery.contradiction_miner import ContradictionMiner

# NEW discovery modules (v2.1)
from discovery.knowledge_gap_detector import KnowledgeGapDetector
from discovery.anomaly_detector import AnomalyDetector
from discovery.missing_variable_generator import MissingVariableGenerator
from discovery.mechanism_generator import MechanismGenerator
from discovery.prediction_engine import PredictionEngine
from discovery.theory_competition import TheoryCompetition
from discovery.discovery_scorer import DiscoveryScorer
from discovery.computational_verification import ComputationalVerifier
from discovery.falsification_engine import FalsificationEngine
from discovery.discovery_tournament import DiscoveryTournament
from discovery.scientific_simulator import ScientificSimulator
from discovery.bayesian_scorer import BayesianScorer
from discovery.literature_contradiction_scorer import LiteratureContradictionScorer
from discovery.resilient_llm import ResilientLLM
from discovery.hypothesis_memory import HypothesisMemory


# Domain detection (Luka — Space Focused)
DOMAIN_KEYWORDS = {
    "space_astronomy": ["phosphine", "venus", "exoplanet", "mars", "jupiter", "black hole",
                        "neutron star", "pulsar", "galaxy", "telescope", "nasa", "jwst",
                        "spectral", "atmosphere", "biosignature", "cosmolog", "megastructure",
                        "dyson", "technosignature", "stellar", "dimming", "cme", "astrophysics"],
    "isro_earth_observation": ["isro", "mosdac", "bhuvan", "insat", "oceansat", "scatsat",
                               "saphir", "monsoon", "cyclone", "bay of bengal", "indian ocean"],
    "solar_system": ["planet", "asteroid", "comet", "moon", "crater", "magnetosphere",
                     "heliosphere", "solar wind", "kuiper belt", "oort cloud"],
    "climate_energy": ["climate", "carbon", "emission", "renewable", "solar", "wind",
                       "greenhouse", "warming", "co2", "temperature", "insolation"],
}


def _filter_contradicted_predictions(accepted_preds: list, validations: list) -> list:
    """Gate: Remove predictions contradicted by literature validation."""
    contradicted_statements = set()
    for v in validations:
        if v.get("validation_status") == "contradicted":
            v_pred = v.get("prediction", "")[:200]
            contradicted_statements.add(v_pred)
            # Also try the full statement
            full = v.get("prediction", "")
            if full:
                contradicted_statements.add(full)
    if not contradicted_statements:
        return accepted_preds
    filtered = []
    for p in accepted_preds:
        statement = p.get("statement", p.get("description", ""))[:200]
        full_statement = p.get("statement", p.get("description", ""))
        if statement in contradicted_statements or full_statement in contradicted_statements:
            continue
        filtered.append(p)
    return filtered or accepted_preds  # never empty all predictions


def detect_domain(topic: str) -> str:
    topic_lower = topic.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in topic_lower)
        if score > 0:
            scores[domain] = score
    return max(scores, key=scores.get) if scores else "general"


# ── Domain-specific search keywords for literature ──
# These are USED to build better search queries, not to filter results.
DOMAIN_SEARCH_KEYWORDS = {
    "solar_system": {
        "enceladus": ["Enceladus plume composition", "tiger stripe fractures tidal heating",
                      "cryovolcanism icy moon", "subsurface ocean Enceladus"],
        "europa": ["Europa subsurface ocean", "plume detection Europa", "tidal flexing ice shell"],
        "titan": ["Titan atmosphere methane", "Titan hydrocarbon lakes", "Titan cryovolcanism"],
        "mars": ["Mars methane detection", "Mars subsurface water", "Mars atmosphere loss"],
        "general": ["icy moon geophysics", "tidal dissipation heating", "cryovolcanism solar system"],
    },
    "space_astronomy": {
        "corona": ["solar corona heating mechanism", "Alfvén wave dissipation corona",
                    "nanoflare heating solar", "coronal loop oscillation damping",
                    "magnetic reconnection corona", "chromosphere transition region heating"],
        "black hole": ["black hole accretion disk", "event horizon photon ring",
                       "Hawking radiation quantum", "gravitational wave merger",
                       "black hole information paradox"],
        "exoplanet": ["exoplanet atmosphere characterization JWST", "transit spectroscopy",
                      "biosignature detection", "exoplanet habitability zone"],
        "neutron star": ["neutron star equation of state", "pulsar timing array",
                         "magnetar giant flare", "neutron star merger kilonova"],
        "dark matter": ["dark matter direct detection", "WIMP nuclear recoil",
                        "dark matter annihilation signal", "SIDM self-interaction cross section"],
        "general": ["stellar evolution nucleosynthesis", "galaxy formation simulation",
                    "cosmic microwave background anisotropy"],
    },
    "physics": {
        "supercon": ["high temperature superconductor mechanism", "Cooper pair formation",
                     "BCS theory cuprate", "spin fluctuation pairing", "Tc superconductor"],
        "quantum": ["quantum entanglement Bell test", "quantum error correction surface code",
                    "decoherence environment interaction", "quantum phase transition"],
        "plasma": ["plasma turbulence cascade", "magnetic reconnection rate",
                   "collisionless shock dissipation", "kinetic Alfvén wave"],
        "general": ["phase transition critical phenomena", "renormalization group flow",
                    "topological defect formation"],
    },
    "drug_discovery": {
        "alzheimer": ["amyloid beta aggregation mechanism", "tau protein phosphorylation",
                      "neuroinflammation microglia", "blood brain barrier drug delivery"],
        "cancer": ["tumor microenvironment immune evasion", "oncogene targeted therapy",
                   "drug resistance mutation", "checkpoint inhibitor combination"],
        "general": ["drug target binding affinity", "ADMET prediction machine learning",
                    "clinical trial failure analysis"],
    },
    "climate_energy": {
        "solar": ["perovskite solar cell stability", "tandem cell efficiency",
                  "photovoltaic degradation mechanism"],
        "climate": ["climate model projection uncertainty", "carbon capture technology",
                    "aerosol radiative forcing"],
        "general": ["grid scale energy storage", "electrocatalyst oxygen evolution"],
    },
}


def extract_search_keywords(topic: str, domain: str) -> list:
    """Extract domain-specific search keywords from a topic.

    Instead of passing the raw topic string to search APIs (which returns garbage),
    we extract proper scientific terms and build targeted queries.

    Returns: list of search query strings (first is most specific, rest are broader)
    """
    topic_lower = topic.lower()
    queries = []

    # 1. Try domain-specific keyword matching
    domain_kws = DOMAIN_SEARCH_KEYWORDS.get(domain, {})
    for keyword, sub_queries in domain_kws.items():
        if keyword in topic_lower:
            queries.extend(sub_queries)

    # 2. If no domain match, extract key scientific terms from topic
    if not queries:
        # Remove common filler words, keep domain terms
        filler_words = {
            "why", "how", "does", "do", "is", "are", "the", "a", "an", "its",
            "that", "this", "which", "what", "when", "where", "despite", "being",
            "than", "from", "with", "for", "into", "about", "between", "through",
            "having", "being", "other", "some", "most", "many", "such", "only",
            "also", "very", "can", "could", "would", "should", "may", "might",
            "not", "but", "and", "or", "if", "then", "so", "yet", "still",
            "millions", "degrees", "hotter", "colder", "larger", "smaller",
            "concentrated", "continuous", "tiny", "huge", "significant",
        }
        key_terms = [w for w in topic.split() if w.lower() not in filler_words and len(w) > 3]

        # Build queries from key terms
        if len(key_terms) >= 2:
            # Most specific: all key terms
            queries.append(" ".join(key_terms[:5]))
            # Medium: first 3 terms
            queries.append(" ".join(key_terms[:3]))
            # Broader: domain + first 2 terms
            queries.append(f"{domain.replace('_', ' ')} {key_terms[0]} {key_terms[1]}")

    # 3. Always include the raw topic as fallback
    queries.insert(0, topic)

    # Deduplicate
    seen = set()
    unique = []
    for q in queries:
        q_lower = q.lower().strip()
        if q_lower not in seen and len(q_lower) > 5:
            seen.add(q_lower)
            unique.append(q)

    return unique[:6]


def extract_entities_from_papers(papers: list, domain: str) -> dict:
    """LLM entity extraction from paper abstracts."""
    paper_text = ""
    for i, p in enumerate(papers[:8], 1):
        paper_text += f"\n[{i}] {p['title']}\n    {p.get('abstract', '')[:300]}\n"

    prompt = f"""Extract scientific entities and relationships from these papers.

Domain: {domain}
Papers:
{paper_text}

Output JSON with exactly this structure:
{{
  "entities": [
    {{"name": "entity name", "type": "celestial_body|mission|telescope|exoplanet|phenomenon|star|galaxy|nebula|asteroid|comet|spacecraft|instrument|orbit|concept", "aliases": []}}
  ],
  "relationships": [
    {{"source": "entity1", "source_type": "type", "relation": "activates|inhibits|causes|treats|enables|uses|produces|detects|measures|correlates_with|associated_with|orbiting|part_of", "target": "entity2", "target_type": "type", "confidence": 0.7}}
  ]
}}

Extract 10-25 entities and 10-30 relationships. Be specific, not generic."""

    try:
        from discovery.json_extract import extract_json
        raw = llm_call(prompt, max_tokens=4096, provider="cerebras")
        if raw:
            result = extract_json(raw, expected_key="entities")
            if isinstance(result, dict):
                return result
    except Exception as e:
        print(f"    Entity extraction failed: {e}")

    entities = []
    for p in papers:
        title = p.get("title", "")
        if title:
            entities.append({"name": title[:60], "type": "concept", "aliases": []})
    return {"entities": entities, "relationships": []}


def build_knowledge_graph(papers: list, domain: str) -> KnowledgeGraph:
    """Build knowledge graph from papers — algorithmic extraction (fast, reliable) + optional LLM."""
    print("    Extracting entities from papers...")

    # Primary: algorithmic extraction (always works, no LLM needed)
    from discovery.entity_extractor import extract_entities_from_papers as algo_extract
    extracted = algo_extract(papers, domain)
    print(f"    Algorithmic: {len(extracted.get('entities', []))} entities, {len(extracted.get('relationships', []))} relationships")

    # Optional enhancement: LLM extraction for additional entities/relationships
    try:
        llm_extracted = extract_entities_from_papers(papers[:5], domain)  # fewer papers for speed
        llm_entities = llm_extracted.get("entities", [])
        llm_rels = llm_extracted.get("relationships", [])
        if llm_entities or llm_rels:
            # Merge LLM results (add non-duplicates)
            existing_names = {e["name"].lower() for e in extracted.get("entities", [])}
            for e in llm_entities:
                if e["name"].lower() not in existing_names:
                    extracted.setdefault("entities", []).append(e)
                    existing_names.add(e["name"].lower())
            for r in llm_rels:
                key = (r.get("source", "").lower(), r.get("target", "").lower())
                if key not in {(rel.get("source", "").lower(), rel.get("target", "").lower()) for rel in extracted.get("relationships", [])}:
                    extracted.setdefault("relationships", []).append(r)
            print(f"    + LLM enhancement: {len(llm_entities)} entities, {len(llm_rels)} relationships")
    except Exception as e:
        print(f"    LLM enhancement skipped: {e}")

    graph = KnowledgeGraph(persist=True)  # Load previous graph — knowledge accumulates across runs
    graph.domain = domain
    # Track run count for staleness management
    graph._run_count = getattr(graph, '_run_count', 0) + 1
    print(f"  [Graph] Loaded {len(graph.entities)} entities, {len(graph.relationships)} relationships from previous runs")

    for p in papers:
        pmid = p.get("id", p.get("citation_key", "unknown"))
        graph.add_paper(pmid, p["title"], p.get("abstract", ""), p.get("url", ""), p.get("year", ""))

    for ent in extracted.get("entities", []):
        pmid = papers[0].get("id", "unknown") if papers else "unknown"
        graph.add_paper_entities([ent], pmid)

    for rel in extracted.get("relationships", []):
        pmid = papers[0].get("id", "unknown") if papers else "unknown"
        graph.add_relationships([rel], pmid)

    # Prune stale entities — keep only those referenced by papers
    if len(graph.entities) > 500:
        current_paper_ids = set(p.get("id", p.get("citation_key", "")) for p in papers)
        stale = []
        for eid, ent in graph.entities.items():
            ent_papers = set(ent.get("papers", []))
            # Keep if referenced by any current paper
            if ent_papers & current_paper_ids:
                continue
            # Keep if referenced by 3+ papers (well-established entity)
            if len(ent_papers) >= 3:
                continue
            # Prune entities with only old paper references
            if len(ent_papers) <= 1:
                stale.append(eid)
        for eid in stale[:200]:  # max 200 pruned per run
            del graph.entities[eid]
        if stale:
            print(f"    Pruned {len(stale)} stale entities (kept {len(graph.entities)})")

    graph.save(session_id=f"discovery_v2_{int(time.time())}")
    print(f"    Graph: {len(graph.entities)} entities, {len(graph.relationships)} relationships")
    return graph


def enrich_knowledge_graph(graph, domain: str):
    """Enrich knowledge graph with external APIs (Luka Space APIs)."""
    from discovery.domains import get_domain
    domain_cfg = get_domain(domain)
    if not domain_cfg:
        return 0

    enrich_sources = domain_cfg.get("enrichment", [])
    if not enrich_sources:
        return 0

    print(f"    Enriching with: {', '.join(enrich_sources)}")
    total_enriched = 0

    # API Map — all enrichment sources
    api_map = {
        "nasa_api": ("discovery.nasa_api", "enrich_entities"),
        "isro_api": ("discovery.isro_api", "enrich_entities"),
        "arxiv": ("discovery.arxiv_api", "enrich_entities"),
        "nasa_power": ("discovery.nasa_power", "enrich_entities"),
        "semantic_scholar": ("discovery.semantic_scholar", "enrich_entities"),
        "simbad_api": ("discovery.simbad_api", "enrich_entities"),
        "vizier_api": ("discovery.vizier_api", "enrich_entities"),
        "iau_api": ("discovery.iau_api", "enrich_entities"),
        "esa_sky_api": ("discovery.esa_sky_api", "enrich_entities"),
        "jpl_horizons_api": ("discovery.jpl_horizons_api", "enrich_entities"),
        "noaa_api": ("discovery.noaa_api", "enrich_entities"),
        "usgs_planetary_api": ("discovery.usgs_planetary_api", "enrich_entities"),
        # New space APIs
        "celestrak_api": ("discovery.celestrak_api", "enrich_entities"),
        "noaa_swpc_api": ("discovery.noaa_swpc_api", "enrich_entities"),
        "nasa_donki_api": ("discovery.nasa_donki_api", "enrich_entities"),
        "mast_api": ("discovery.mast_api", "enrich_entities"),
        "ligo_gwosc_api": ("discovery.ligo_gwosc_api", "enrich_entities"),
        "jpl_smallbody_api": ("discovery.jpl_smallbody_api", "enrich_entities"),
        "mpc_api": ("discovery.mpc_api", "enrich_entities"),
        "gcn_api": ("discovery.gcn_api", "enrich_entities"),
    }

    for source in enrich_sources:
        if source in api_map:
            module_path, func_name = api_map[source]
            try:
                import importlib
                module = importlib.import_module(module_path)
                func = getattr(module, func_name)
                if source == "arxiv":
                    count = func(graph, query=domain_cfg.get("label", ""))
                else:
                    count = func(graph)
                total_enriched += count
                print(f"      + {source}: {count} enriched")
            except (ImportError, AttributeError, Exception):
                pass

    return total_enriched

# ══════════════════════════════════════════════════════════════════
# PHASE REGISTRY INFRASTRUCTURE
# The pipeline is no longer one 1,500-line function: every phase is a
# standalone function `_phase_<name>(ctx)` operating on a PipelineContext,
# and PHASE_REGISTRY declares the execution order + mode gating.
#
# Adding a new phase = write `def _phase_new(ctx)` + one registry entry.
# Mode gating is declarative (min_mode), not an early-return hardcode.
# ══════════════════════════════════════════════════════════════════
import os
import re as _re_module

from discovery.scoring_config import (
    LLM_MAX_PROMPT_CHARS,
    LLM_TRUNCATION_HEAD_FRACTION,
    CV_ROBUSTNESS_THRESHOLD,
    CV_ROBUSTNESS_PENALTY_SCALE,
    CV_ROBUSTNESS_BONUS_SCALE,
    UNIFIED_DISCOVERY_WEIGHT,
    UNIFIED_REFINEMENT_WEIGHT,
    UNIFIED_FLAG_RATIO,
    UNIFIED_FLAG_CAP_MARGIN,
    UNIFIED_OVERRIDE_RATIO,
    UNIFIED_WEAKNESS_REFINEMENT_SCORE,
    grade_for,
    cross_validation_adjustment,
    unify_scores,
)

# Quick mode = foundation phases only; standard adds analysis phases;
# full adds everything (tournament, verification, refinement, reflexion).
MODE_LEVELS = {"quick": 1, "standard": 2, "full": 3}

# Phase-aware LLM routing — spread work across ALL providers.
# Each provider has internal fallback: OpenRouter → NVIDIA → Cerebras.
PHASE_PROVIDERS = {
    "literature": "openrouter",       # Phase 1: extraction — OpenRouter fast
    "knowledge_graph": "openrouter",  # Phase 2: entity extraction — OpenRouter fast
    "gap_detection": "openrouter",    # Phase 3: pattern matching — OpenRouter fast
    "anomaly_detection": "openrouter",# Phase 4: pattern matching — OpenRouter fast
    "missing_variables": "cerebras",  # Phase 5: creative hypotheses — Cerebras fast
    "mechanism_generation": "nvidia", # Phase 6: deep causal reasoning — Nemotron 3 Ultra
    "prediction_engine": "nvidia",    # Phase 7: testable predictions — Nemotron 3 Ultra
    "theory_competition": "openrouter", # Phase 8: tournament gen — OpenRouter volume
    "adversarial_test": "cerebras",   # Phase 8.5: critique — Cerebras fast
    "critical_evaluation": "nvidia",  # Phase 8.6: formal assessment — Nemotron 3 Ultra
    "skeptic_review": "openrouter",   # Phase 11: adversarial review — OpenRouter
    "scoring": "openrouter",          # Phase 12: mostly algorithmic — OpenRouter fast
    "self_improvement": "openrouter", # Phase 15: reflective analysis — OpenRouter
    "gap_discovery": "cerebras",      # Phase 16: creative gap detection — Cerebras fast
}

# Mutable so closures below stay cheap; one pipeline runs per process.
CURRENT_PHASE = {"name": "missing_variables"}


def _truncated_llm(prompt, max_tokens=4096, **kwargs):
    """LLM call with fused head+tail truncation + phase-aware routing + multi-layer fallback.

    The truncation fuse only fires for run-away prompts (default 20,000 chars,
    override with LUKA_MAX_PROMPT_CHARS). The head is kept (it holds the task
    definition); the tail is kept too (it usually holds the output JSON spec).
    Every truncation is announced — no more silent context loss.
    """
    original_len = len(prompt)
    if original_len > LLM_MAX_PROMPT_CHARS:
        head = int(LLM_MAX_PROMPT_CHARS * LLM_TRUNCATION_HEAD_FRACTION)
        marker = f"\n\n[... context truncated: {original_len} chars → ~{LLM_MAX_PROMPT_CHARS} ...]\n\n"
        tail_chars = max(0, LLM_MAX_PROMPT_CHARS - head - len(marker))
        prompt = prompt[:head] + marker + (prompt[-tail_chars:] if tail_chars else "")
        print(f"    [LLM] Prompt truncated: {original_len} chars → ~{LLM_MAX_PROMPT_CHARS}", flush=True)
    phase = kwargs.get("phase", CURRENT_PHASE["name"])
    provider = PHASE_PROVIDERS.get(phase, "auto")

    try:
        r = llm_call(prompt, json_mode=False, max_tokens=max_tokens, provider=provider)
        if r:
            return r

        if provider != "auto":
            print(f"    [DEBUG] {provider} exhausted, trying auto...", flush=True)
            r = llm_call(prompt, json_mode=False, max_tokens=max_tokens, provider="auto")
            if r:
                return r

        import discovery.llm_client as _lc
        soonest = min(
            getattr(_lc, '_openrouter_cooldown_until', 0),
            getattr(_lc, '_nvidia_cooldown_until', 0),
            getattr(_lc, '_cerebras_cooldown_until', 0),
        )
        wait = max(0, min(20, soonest - time.time()))
        if wait > 0:
            print(f"    [DEBUG] All providers cooling down, waiting {wait:.0f}s...", flush=True)
            time.sleep(wait + 1)
        r = llm_call(prompt, json_mode=False, max_tokens=max_tokens, provider="auto")
        if r:
            return r

        print(f"    [DEBUG] LLM exhausted (all providers)", flush=True)
        return None
    except Exception as e:
        print(f"    [ERROR] LLM exception: {e}", flush=True)
        return None


class PipelineContext:
    """Shared state bag passed through every phase function."""

    def __init__(self, topic: str, domain: str, mode: str,
                 curiosity_constraint: dict, original_topic: str, provider_name: str):
        self.topic = topic
        self.domain = domain
        self.mode = mode
        self.curiosity_constraint = curiosity_constraint  # None for Track A, set for Track B
        self.original_topic = original_topic
        self.provider_name = provider_name
        self.archive_context = ""
        self.hyp_mem = None
        self.run_id = f"run_{int(time.time())}"
        self.t_start = time.time()
        self.report = {
            "topic": topic,
            "domain": domain,
            "mode": mode,
            "pipeline_version": "v2_discovery",
            "started_at": datetime.now().isoformat(),
            "phases": {},
            "errors": [],
        }
        # Phase outputs — stay None/empty until the owning phase fills them
        self.papers = []
        self.citation_context = ""
        self.graph = None
        self.gaps = []
        self.anomalies = []
        self.hidden_variables = []
        self.mechanisms = []
        self.predictions = []
        self.accepted_preds = []
        self.theories = []
        self.winner = None
        self.all_contradictions = []
        self.skeptic_result = {}


# ══════════════════════════════════════════════════════════════════
# PHASE FUNCTIONS (execution order == registry order below)
# ══════════════════════════════════════════════════════════════════

def _phase_literature(ctx):
    print("\n[Phase 1/16] LITERATURE — Fetching papers from 4 sources (domain-aware)...", flush=True)
    # Extract domain-specific search keywords (not generic topic string)
    search_queries = extract_search_keywords(ctx.topic, ctx.domain)
    print(f"  Search queries: {search_queries[:3]}")

    # Round 1: Most specific domain-aware query
    papers = fetch_papers(search_queries[0], max_arxiv=20, max_pubmed=20, max_s2=20, max_crossref=20)
    print(f"  Round 1 ({search_queries[0][:50]}): {len(papers)} papers")

    # Round 2: Additional domain-aware queries (broader terms)
    existing_titles = {p["title"].lower()[:60] for p in papers}
    for qi, query in enumerate(search_queries[1:4], 2):
        r = fetch_papers(query, max_arxiv=12, max_pubmed=12, max_s2=12, max_crossref=12)
        added = 0
        for p in r:
            if p["title"].lower()[:60] not in existing_titles:
                existing_titles.add(p["title"].lower()[:60])
                papers.append(p)
                added += 1
        print(f"  Round {qi} ({query[:50]}): +{added} new papers ({len(papers)} total)")

    # Round 3: If still low, use remaining queries
    if len(papers) < 40 and len(search_queries) > 4:
        for query in search_queries[4:]:
            r = fetch_papers(query, max_arxiv=10, max_pubmed=10, max_s2=10, max_crossref=10)
            added = 0
            for p in r:
                if p["title"].lower()[:60] not in existing_titles:
                    existing_titles.add(p["title"].lower()[:60])
                    papers.append(p)
                    added += 1
            print(f"  Round extra ({query[:50]}): +{added} new papers")

    sources = set(p['source'] for p in papers)
    print(f"  Total: {len(papers)} papers from {len(sources)} sources: {', '.join(sorted(sources))}")
    for p in papers[:5]:
        cites = p.get('citation_count', '')
        cite_str = f" ({cites} citations)" if cites else ""
        print(f"    [{p['source']}] {p['title'][:65]}{cite_str}")
    ctx.papers = papers
    ctx.citation_context = build_citation_context(papers)
    ctx.report["phases"]["literature"] = {
        "papers_found": len(papers),
        "sources": list(set(p["source"] for p in papers)),
        "papers": papers,
    }


def _phase_citation_network(ctx):
    print("\n[Phase 1.5/16] CITATION NETWORK — 2-hop citation walk...", flush=True)
    from discovery.citation_grounding import traverse_citation_network
    ctx.papers = traverse_citation_network(ctx.papers, hop_depth=2, top_n=5, refs_per_paper=10)
    ctx.citation_context = build_citation_context(ctx.papers)
    ctx.report["phases"]["citation_network"] = {
        "total_papers": len(ctx.papers),
        "citation_hop1": len([p for p in ctx.papers if p.get("source") == "citation_hop1"]),
        "citation_hop2": len([p for p in ctx.papers if p.get("source") == "citation_hop2"]),
    }


def _phase_knowledge_graph(ctx):
    print("\n[Phase 2/16] KNOWLEDGE GRAPH — Building...", flush=True)
    ctx.graph = build_knowledge_graph(ctx.papers, ctx.domain)
    ctx.report["phases"]["knowledge_graph"] = {
        "entities": len(ctx.graph.entities),
        "relationships": len(ctx.graph.relationships),
        "papers": len(ctx.graph.papers),
    }


def _phase_enrichment(ctx):
    print("\n[Phase 2.5/16] ENRICHMENT — Calling external APIs...", flush=True)
    enriched_count = enrich_knowledge_graph(ctx.graph, ctx.domain)
    ctx.report["phases"]["enrichment"] = {
        "enriched_entities": enriched_count
    }


def _phase_graph_inference(ctx):
    print("\n[Phase 2.6/16] GRAPH INFERENCE — Reducing fragmentation...", flush=True)
    from discovery.graph_inferrer import reduce_fragmentation
    frag_result = reduce_fragmentation(ctx.graph, min_cooccurrence=2)
    print(f"  Relationships added: {frag_result.get('relationships_added', 0)}")
    print(f"  Components: {frag_result.get('components_before', 0)} -> {frag_result.get('components_after', 0)}")
    ctx.report["phases"]["graph_inference"] = frag_result


def _phase_gap_detection(ctx):
    print("\n[Phase 3/16] GAP DETECTION — Finding what's missing...", flush=True)
    # Use algorithmic gap detection (fast, reliable, no LLM needed)
    gap_detector = KnowledgeGapDetector(graph=ctx.graph, llm_call=None)
    gap_results = gap_detector.detect_gaps(ctx.topic, ctx.domain)
    gaps = gap_results.get("top_gaps", [])
    print(f"  Found {len(gaps)} knowledge gaps")
    for g in gaps[:3]:
        print(f"    [{g.get('type', '?')}] {g.get('reason', '')[:80]}")
    ctx.report["phases"]["gap_detection"] = gap_results.get("summary", {})
    ctx.report["phases"]["gap_detection"]["top_gaps"] = gaps
    ctx.gaps = gaps


def _phase_literature_refinement(ctx):
    print("\n[Phase 3.5/16] LITERATURE REFINEMENT — Targeted search based on gaps...", flush=True)
    # Adaptive literature search: analyze what was found, generate
    # targeted queries for gaps, fetch again. Multi-round approach.
    gaps = ctx.gaps
    papers = ctx.papers
    graph = ctx.graph
    topic = ctx.topic
    # Generate targeted queries from gaps
    gap_queries = set()
    for g in gaps[:6]:
        if isinstance(g, dict):
            reason = g.get("reason", g.get("description", ""))
            gtype = g.get("type", "")
            # Extract key terms from gap descriptions
            words = [w for w in reason.split() if len(w) > 4 and w.lower() not in
                     ("these", "their", "there", "which", "about", "would", "could",
                      "should", "between", "through", "having", "being", "other")]
            if len(words) >= 2:
                gap_queries.add(" ".join(words[:4]))

    # Also generate queries from graph entities that have few connections
    if hasattr(graph, 'entities') and graph.entities:
        entity_degrees = {}
        for r in graph.relationships:
            if isinstance(r, dict):
                src = r.get("source", "")
                tgt = r.get("target", "")
            else:
                src, tgt = r[0], r[1]
            entity_degrees[src] = entity_degrees.get(src, 0) + 1
            entity_degrees[tgt] = entity_degrees.get(tgt, 0) + 1
        # Find under-connected entities
        low_degree = sorted(entity_degrees.items(), key=lambda x: x[1])[:5]
        for eid, deg in low_degree:
            ename = graph.entities.get(eid, {}).get("name", eid)
            if ename and len(ename) > 3:
                gap_queries.add(f"{topic} {ename}")

    if gap_queries:
        existing_titles = {p["title"].lower()[:60] for p in papers}
        new_papers = []
        for query in list(gap_queries)[:3]:
            try:
                round_papers = fetch_papers(query, max_arxiv=8, max_pubmed=8, max_s2=8)
                for p in round_papers:
                    if p["title"].lower()[:60] not in existing_titles:
                        existing_titles.add(p["title"].lower()[:60])
                        new_papers.append(p)
            except Exception:
                continue

        if new_papers:
            papers.extend(new_papers)
            # Rebuild graph with new papers
            try:
                for p in new_papers:
                    graph.add_entity(p.get("title", "")[:80], entity_type="paper")
            except Exception:
                pass
            ctx.papers = papers
            print(f"  Gap-targeted search: +{len(new_papers)} new papers (total: {len(papers)})")
        else:
            print("  Gap-targeted search: no new papers found")
    else:
        print("  No gap queries generated — skipping refinement")


def _phase_anomaly_detection(ctx):
    print("\n[Phase 4/16] ANOMALY DETECTION — Finding what doesn't fit...", flush=True)
    # Use algorithmic anomaly detection (fast, reliable)
    anomaly_detector = AnomalyDetector(graph=ctx.graph, llm_call=None)
    anomaly_results = anomaly_detector.detect_anomalies(ctx.topic, ctx.domain)
    anomalies = anomaly_results.get("top_anomalies", [])
    print(f"  Found {len(anomalies)} anomalies")
    for a in anomalies[:3]:
        print(f"    [{a.get('type', '?')}] {a.get('reason', '')[:80]}")
    ctx.report["phases"]["anomaly_detection"] = anomaly_results.get("summary", {})
    ctx.report["phases"]["anomaly_detection"]["top_anomalies"] = anomalies
    ctx.anomalies = anomalies


def _phase_curious_questioning(ctx):
    print("\n[Phase 0/16] CURIOUS QUESTIONING — The Newton Step...", flush=True)
    from discovery.curious_questioning import CuriousQuestioning
    cq = CuriousQuestioning(llm_call=_truncated_llm)
    curious_results = cq.run(ctx.topic, ctx.domain, papers=ctx.papers)
    core_q = curious_results.get("reframed", "")
    n_obs = len(curious_results.get("observations", []))
    n_qs = len(curious_results.get("questions", []))
    n_gen = len(curious_results.get("generalizations", []))
    print(f"  Observations: {n_obs}, Questions: {n_qs}, Generalizations: {n_gen}")
    if core_q:
        print(f"  Core question: {core_q[:80]}")
    ctx.report["phases"]["curious_questioning"] = {
        "observations": n_obs,
        "questions": n_qs,
        "generalizations": n_gen,
        "core_question": core_q,
        "constraint": curious_results.get("constraint"),
    }


def _phase_missing_variables(ctx):
    print("\n[Phase 5/16] MISSING VARIABLES — Proposing hidden factors...", flush=True)
    CURRENT_PHASE["name"] = "missing_variables"
    mv_generator = MissingVariableGenerator(graph=ctx.graph, llm_call=_truncated_llm)
    mv_results = mv_generator.generate(ctx.gaps, ctx.anomalies, ctx.topic, ctx.domain, ctx.papers, ctx.archive_context)
    hidden_variables = mv_results.get("hidden_variables", [])

    # Algorithmic fallback if LLM returns nothing
    if not hidden_variables:
        from discovery.algorithmic_discovery import generate_hypotheses_algorithmic
        hidden_variables = generate_hypotheses_algorithmic(ctx.graph, ctx.gaps, ctx.anomalies, ctx.topic, ctx.domain)
        if hidden_variables:
            print(f"  Algorithmic fallback: {len(hidden_variables)} hypotheses")

    # Validate hidden variables — reject those with unjustified parameters
    validated_hvs = []
    for hv in hidden_variables:
        name = hv.get("name", "")
        desc = hv.get("description", "")
        # Skip generic "Hidden mechanism connecting X" without explanation
        if "Hidden mechanism connecting" in name and len(desc) < 100:
            continue
        # Require key_parameters to have origin/measurement
        params = hv.get("key_parameters", [])
        for p in params:
            if isinstance(p, dict) and not p.get("origin") and not p.get("units"):
                p["status"] = "unjustified"
                p["penalty"] = -0.1
        validated_hvs.append(hv)
    hidden_variables = validated_hvs

    print(f"  Proposed {len(hidden_variables)} hidden variables")
    for hv in hidden_variables[:3]:
        print(f"    [{hv.get('type', '?')}] {hv.get('name', '?')}")
        print(f"      {hv.get('description', '')[:100]}")
    ctx.report["phases"]["missing_variables"] = {
        "proposed": len(hidden_variables),
        "variables": [hv.get("name", "?") for hv in hidden_variables],
        "variable_details": hidden_variables,
    }
    ctx.hidden_variables = hidden_variables


def _phase_mechanism_generation(ctx):
    print("\n[Phase 6/16] MECHANISM GENERATION — Building causal pathways...", flush=True)
    CURRENT_PHASE["name"] = "mechanism_generation"
    mech_generator = MechanismGenerator(graph=ctx.graph, llm_call=_truncated_llm)
    mech_results = mech_generator.generate_mechanisms(
        ctx.hidden_variables, ctx.gaps, ctx.anomalies, ctx.topic, ctx.domain, ctx.papers, ctx.archive_context,
        constraint=ctx.curiosity_constraint
    )
    mechanisms = mech_results.get("mechanisms", [])

    # Algorithmic fallback
    if not mechanisms:
        from discovery.algorithmic_discovery import generate_mechanisms_algorithmic
        mechanisms = generate_mechanisms_algorithmic(ctx.graph, ctx.hidden_variables, ctx.topic, ctx.domain)
        if mechanisms:
            print(f"  Algorithmic fallback: {len(mechanisms)} mechanisms")

    # Validate mechanisms — allow single-step direct observations
    validated_mechanisms = []
    for m in mechanisms:
        name = m.get("name", "")
        steps = m.get("steps", [])
        # Reject generic "Hidden mechanism connecting X" without details
        if "Hidden mechanism connecting" in name and len(steps) < 1:
            continue
        # Allow single-step mechanisms (direct observations are valid)
        if len(steps) < 1:
            continue
        # Tag single-step as direct_observation
        if len(steps) == 1:
            m.setdefault("type", "direct_observation")
        # Mark as speculative if missing key fields — no confidence cap
        if not m.get("inputs") and not m.get("outputs"):
            m["status"] = "speculative"
            # Don't cap confidence — let scorer decide
        validated_mechanisms.append(m)
    mechanisms = validated_mechanisms

    print(f"  Generated {len(mechanisms)} mechanisms")
    for m in mechanisms[:3]:
        print(f"    [{m.get('type', '?')}] {m.get('name', '?')}")
        for s in m.get("steps", [])[:2]:
            s_str = str(s)[:80] if s else ""
            print(f"      → {s_str}")
    ctx.report["phases"]["mechanism_generation"] = {
        "mechanisms_generated": len(mechanisms),
        "types": list(set(m.get("type", "?") for m in mechanisms)),
        "mechanism_details": mechanisms,
    }
    ctx.mechanisms = mechanisms


def _phase_derivation_validator(ctx):
    print("\n[Phase 6.4/16] DERIVATION VALIDATOR — Checking reproducibility...", flush=True)
    from discovery.derivation_validator import validate_batch
    dv_result = validate_batch(ctx.mechanisms)
    n_reproducible = dv_result.get("reproducible", 0)
    n_total = dv_result.get("total", 0)
    avg_score = dv_result.get("avg_score", 0)
    print(f"  Reproducible: {n_reproducible}/{n_total}, Avg score: {avg_score:.2f}")
    ctx.report["phases"]["derivation_validation"] = {
        "total": n_total,
        "reproducible": n_reproducible,
        "not_reproducible": dv_result.get("not_reproducible", 0),
        "avg_score": avg_score,
    }


def _phase_math_chain(ctx):
    print("\n[Phase 6.5/16] MATH CHAIN — Deriving equations for mechanisms...", flush=True)
    try:
        from discovery.math_chain import MathChain
        mc = MathChain(llm_call=_truncated_llm)
        ctx.mechanisms = mc.process_mechanisms(ctx.mechanisms, ctx.topic, ctx.domain)
        print(f"  Processed {len(ctx.mechanisms)} mechanisms through math chain")
    except Exception as e:
        # Math chain failure is non-fatal (mechanisms pass through unchanged)
        print(f"  [WARN] Math chain failed: {e}")

    print("\n[Phase 6.6/16] MECHANISM COMPLETENESS — Checking derivational rigor...", flush=True)
    try:
        from discovery.mechanism_completeness import MechanismCompletenessChecker
        checker = MechanismCompletenessChecker(llm_call=_truncated_llm)
        completeness = checker.check_mechanisms(ctx.mechanisms, domain=ctx.domain)
        n_complete = completeness.get("complete", 0)
        n_incomplete = completeness.get("incomplete", 0)
        avg_score = completeness.get("avg_completeness", 0)
        print(f"  Complete: {n_complete}, Incomplete: {n_incomplete}, Avg score: {avg_score:.2f}")
        ctx.report["phases"]["mechanism_completeness"] = {
            "total": completeness.get("total", 0),
            "complete": n_complete,
            "incomplete": n_incomplete,
            "avg_completeness": avg_score,
        }
    except Exception as e:
        print(f"  [WARN] Mechanism completeness check failed: {e}")


def _safe_str_items(items: list) -> list:
    """Defensive conversion: all dict values → strings if not list/dict."""
    safe_items = []
    for item in (items or []):
        if isinstance(item, dict):
            safe = {k: str(v) if not isinstance(v, (list, dict)) else v for k, v in item.items()}
            safe_items.append(safe)
        else:
            safe_items.append(item)
    return safe_items


def _phase_prediction_engine(ctx):
    print("\n[Phase 7/16] PREDICTION ENGINE — Generating testable predictions...", flush=True)
    CURRENT_PHASE["name"] = "prediction_engine"
    # Defensive: convert all mechanism/HV/anomaly values to strings to prevent float+str errors
    safe_mechanisms = _safe_str_items(ctx.mechanisms)
    safe_hvs = _safe_str_items(ctx.hidden_variables)
    safe_anomalies = _safe_str_items(ctx.anomalies)

    pred_engine = PredictionEngine(graph=ctx.graph, llm_call=_truncated_llm)
    try:
        pred_results = pred_engine.generate_predictions(
            safe_mechanisms, safe_hvs, ctx.topic, ctx.domain, safe_anomalies
        )
    except TypeError as te:
        import traceback
        print(f"  [ERROR] Prediction generation failed: {te}", flush=True)
        traceback.print_exc()
        pred_results = {"predictions": [], "prediction_chains": [], "coverage_score": 0.0}
    predictions = pred_results.get("predictions", [])

    # Algorithmic fallback
    if not predictions:
        from discovery.algorithmic_discovery import generate_predictions_algorithmic
        predictions = generate_predictions_algorithmic(ctx.mechanisms, ctx.hidden_variables, ctx.topic, ctx.domain, ctx.graph)
        if predictions:
            print(f"  Algorithmic fallback: {len(predictions)} predictions")

    # Validate predictions
    validation = pred_engine.validate_predictions(predictions)
    accepted_preds = validation.get("accepted", predictions)

    print(f"  Generated {len(predictions)} predictions ({len(accepted_preds)} accepted)")
    for p in accepted_preds[:3]:
        print(f"    [{p.get('type', '?')}] {p.get('statement', '')[:80]}")
    ctx.report["phases"]["prediction_engine"] = {
        "total_predictions": len(predictions),
        "accepted": len(accepted_preds),
        "acceptance_rate": validation.get("acceptance_rate", 0),
        "accepted_details": accepted_preds,
        "predictions": predictions,
        "all_predictions": predictions,
    }
    ctx.predictions = predictions
    ctx.accepted_preds = accepted_preds


def _phase_observability(ctx):
    print("\n[Phase 7.5/16] OBSERVABILITY CHECK — Can we measure these predictions?...", flush=True)
    from discovery.observability_checker import ObservabilityChecker
    obs_checker = ObservabilityChecker(llm_call=_truncated_llm)
    obs_result = obs_checker.check_predictions(ctx.accepted_preds, domain=ctx.domain)
    n_accessible = obs_result.get("accessible", 0)
    n_total = obs_result.get("total", 0)
    rate = obs_result.get("accessibility_rate", 0)
    print(f"  Accessible: {n_accessible}/{n_total} ({rate:.0%})")
    ctx.report["phases"]["observability"] = {
        "total": n_total,
        "accessible": n_accessible,
        "inaccessible": obs_result.get("inaccessible", 0),
        "accessibility_rate": rate,
    }


def _phase_prediction_literature_validation(ctx):
    print("\n[Phase 7.6/16] PREDICTION LITERATURE VALIDATION — Checking against published data...", flush=True)
    from discovery.prediction_literature_validator import PredictionLiteratureValidator
    plv = PredictionLiteratureValidator(llm_call=_truncated_llm)
    plv_result = plv.validate_predictions(ctx.accepted_preds, mechanisms=ctx.mechanisms,
                                          topic=ctx.topic, domain=ctx.domain, existing_papers=ctx.papers)
    summary = plv_result.get("summary", {})
    grounding = summary.get("overall_grounding", "none")
    supported = summary.get("supported", 0)
    contradicted = summary.get("contradicted", 0)
    print(f"  Grounding: {grounding}, Supported: {supported}, Contradicted: {contradicted}")

    # GATE: Remove contradicted predictions from accepted list
    if contradicted > 0 and ctx.accepted_preds:
        validations = plv_result.get("validations", [])
        preds_before = len(ctx.accepted_preds)
        ctx.accepted_preds = _filter_contradicted_predictions(ctx.accepted_preds, validations)
        removed = preds_before - len(ctx.accepted_preds)
        if removed > 0:
            print(f"  [GATE] Removed {removed} contradicted predictions ({preds_before} → {len(ctx.accepted_preds)})")

    ctx.report["phases"]["prediction_literature_validation"] = {
        "overall_grounding": grounding,
        "supported": supported,
        "partially_supported": summary.get("partially_supported", 0),
        "contradicted": contradicted,
        "unverifiable": summary.get("unverifiable", 0),
        "avg_confidence_adjustment": summary.get("avg_confidence_adjustment", 0),
    }


def _phase_theory_competition(ctx):
    print("\n[Phase 8/16] THEORY COMPETITION — Comparing explanations...", flush=True)
    CURRENT_PHASE["name"] = "theory_competition"
    competition = TheoryCompetition(graph=ctx.graph, llm_call=_truncated_llm)
    comp_results = competition.compete(
        ctx.mechanisms, ctx.hidden_variables, ctx.anomalies, ctx.gaps, ctx.topic, ctx.domain, ctx.papers,
        archive_context=ctx.archive_context
    )
    theories = comp_results.get("theories", [])
    winner = comp_results.get("winner")

    # Algorithmic fallback
    if not theories:
        from discovery.algorithmic_discovery import compare_theories_algorithmic
        theories = compare_theories_algorithmic(ctx.hidden_variables, ctx.graph, ctx.papers)
        winner = theories[0] if theories else None
        # Apply remap + metadata extraction to algorithmic theories too
        for t in theories:
            t.setdefault("explains", [])
            t.setdefault("fails_to_explain", [])
            t.setdefault("predictions", [])
            t.setdefault("key_assumptions", [])
            t.setdefault("key_parameters", [])
            t.setdefault("hidden_variables", [])
            t.setdefault("description", "")
            t.setdefault("mechanism", "")
            competition._remap_llm_fields(t)
            competition._extract_metadata_from_description(t)
        if theories:
            print(f"  Algorithmic fallback: {len(theories)} theories compared")
        comp_results["theories_compared"] = len(theories)
        comp_results["eliminated"] = []

    # Validate theories — require causal chains, not just correlations
    validated_theories = []
    for t in theories:
        name = t.get("name", "")
        desc = t.get("description", t.get("mechanism", ""))
        # Track causal status for reporting — don't penalize
        # Correlations are valid starting points (Darwin didn't know DNA)
        if "correlat" in desc.lower() and "caus" not in desc.lower():
            t["causal_status"] = "correlation_only"
            t["causal_penalty"] = 0  # no penalty — correlations are valid
        has_causal = any(kw in desc.lower() for kw in ["causes", "leads to", "produces", "generates", "triggers", "mediates", "drives"])
        if not has_causal:
            t["causal_status"] = "no_causal_claim"
            t["causal_penalty"] = 0  # no penalty — track for reporting only
        validated_theories.append(t)
    theories = validated_theories

    print(f"  {len(theories)} competing theories (tournament)")
    if winner:
        w_score = winner.get("scores", {}).get("overall", 0)
        print(f"  Winner: {winner.get('name', '?')} (score: {w_score:.2f})")
    eliminated = comp_results.get("eliminated", [])
    if eliminated:
        print(f"  Eliminated: {len(eliminated)} theories killed in tournament")
    ctx.report["phases"]["theory_competition"] = {
        "theories_compared": comp_results.get("theories_compared", len(theories)),
        "theories": theories,
        "eliminated": eliminated,
        "winner": winner,
        "winner_name": winner.get("name") if winner else None,
        "winner_score": winner.get("scores", {}).get("overall", 0) if winner else 0,
        "tournament_log": comp_results.get("tournament_log", []),
        "competition_analysis": comp_results.get("competition_analysis", ""),
        "discriminating_experiments": comp_results.get("discriminating_experiments", []),
    }
    ctx.theories = theories
    ctx.winner = winner


def _apply_adversarial_verdicts(ctx):
    """Gate: adversarial-test verdicts now actually act on the theory pool.

    - 'killed' theories are REMOVED from the active pool (they used to just
      be reported and then silently scored/finalized as if nothing happened).
    - 'weakened' theories are flagged so later scoring/printing can trust them less.
    - If the tournament winner got killed, the best surviving theory takes over.
    """
    test_results = ctx.report["phases"].get("adversarial_test") or {}
    theory_tests = test_results.get("theory_tests", []) or []
    if not theory_tests or not ctx.theories:
        return

    def name_key(t):
        return str(t.get("name", "")).strip().lower()

    verdict_by_name = {}
    for tt in theory_tests:
        n = str(tt.get("name", "")).strip().lower()
        if n:
            verdict_by_name[n] = tt

    def verdict_of(t):
        return (verdict_by_name.get(name_key(t)) or {}).get("verdict", "")

    killed = [t for t in ctx.theories if verdict_of(t) == "killed"]
    survivors = [t for t in ctx.theories if verdict_of(t) != "killed"]

    if survivors:
        for t in survivors:
            v = verdict_of(t)
            t["adversarial_verdict"] = v if v in ("weakened", "survived") else "untouched"
            if v == "weakened":
                t["scores"] = t.get("scores", {})
                t["scores"]["adversarial_weakened"] = True

    if not killed:
        return

    if not survivors:
        # Adversary killed everything — trust the tournament ranking instead of
        # wiping the report, but record what happened loudly.
        test_results["elimination_note"] = ("all_theories_killed_kept_tournament_ranking")
        print("  [GATE] Adversarial test killed every theory — verdicts recorded, "
              "tournament ranking retained", flush=True)
        return

    for t in killed:
        tt = verdict_by_name.get(name_key(t)) or {}
        t["eliminated_by"] = "adversarial_test"
        reason = tt.get("reasoning") or tt.get("falsification") or "killed by adversarial test"
        t["elimination_reason"] = str(reason)[:300]

    test_results["eliminated_theories"] = [
        {"name": t.get("name", "?"), "reason": t.get("elimination_reason", "")} for t in killed
    ]

    print(f"  [GATE] Adversarial elimination: {len(killed)} killed theory(ies) removed from "
          f"the active pool ({len(ctx.theories)} → {len(survivors)})", flush=True)

    # Winner replacement: if the crowned theory was killed, promote best survivor
    if ctx.winner and verdict_of(ctx.winner) == "killed":
        best = max(survivors, key=lambda t: t.get("scores", {}).get("overall", 0))
        print(f"  [GATE] Winner '{ctx.winner.get('name', '?')}' was killed — promoting "
              f"'{best.get('name', '?')}' (score: {best.get('scores', {}).get('overall', 0)})", flush=True)
        ctx.winner = best

    ctx.theories = survivors
    tc_phase = ctx.report["phases"].get("theory_competition") or {}
    tc_phase["theories"] = survivors
    tc_phase["winner"] = ctx.winner
    tc_phase["winner_name"] = ctx.winner.get("name") if ctx.winner else None
    tc_phase["winner_score"] = ctx.winner.get("scores", {}).get("overall", 0) if ctx.winner else 0
    ctx.report["phases"]["theory_competition"] = tc_phase


def _phase_adversarial_test(ctx):
    print("\n[Phase 8.5/16] ADVERSARIAL TEST — Attacking every discovery...", flush=True)
    CURRENT_PHASE["name"] = "adversarial_test"
    from discovery.test_stage import AdversarialTest
    tester = AdversarialTest(llm_call=_truncated_llm)
    test_results = tester.test_discoveries(
        ctx.hidden_variables, ctx.mechanisms, ctx.theories, ctx.papers, ctx.topic, ctx.domain, ctx.gaps, ctx.anomalies
    )
    ctx.report["phases"]["adversarial_test"] = test_results

    # Print results
    summary = test_results.get("survival_summary", {})
    for category in ["hidden_variables", "mechanisms", "theories"]:
        s = summary.get(category, {})
        survived = s.get("survived", 0)
        weakened = s.get("weakened", 0)
        killed = s.get("killed", 0)
        total = s.get("started", 0)
        print(f"  {category}: {total} tested → {survived} survived, {weakened} weakened, {killed} killed")

    # Print individual verdicts
    for test_list_name in ["hidden_variable_tests", "mechanism_tests", "theory_tests"]:
        for t in test_results.get(test_list_name, [])[:3]:
            name = t.get("name", "?")
            verdict = t.get("verdict", "?")
            score = t.get("attack_score", 0)
            existing = t.get("existing_theory", "")
            falsification = t.get("falsification", "")
            icon = {"survived": "✓", "weakened": "⚠", "killed": "✗"}.get(verdict, "?")
            print(f"    {icon} {name[:40]} ({verdict}, score: {score:.2f})")
            if existing and existing != "None known":
                print(f"      Existing theory: {str(existing)[:80]}")
            if falsification:
                print(f"      Falsification: {str(falsification)[:80]}")

    # NEW GATE: enforce the verdicts on the actual pool
    _apply_adversarial_verdicts(ctx)


def _phase_peer_review(ctx):
    print("\n[Phase 8.6/16] CRITICAL EVALUATION — Formal assessment...", flush=True)
    CURRENT_PHASE["name"] = "critical_evaluation"
    # Evaluates the top discovery across 6 dimensions:
    # novelty, methodology, significance, clarity, limitations, reproducibility.
    from discovery.peer_review import CriticalEvaluator
    reviewer = CriticalEvaluator(llm_call=_truncated_llm)
    top_theory = ctx.winner or (ctx.theories[0] if ctx.theories else {})
    if top_theory:
        _phases = ctx.report.get("phases", {})
        peer_review = reviewer.review(
            top_theory, ctx.mechanisms, ctx.accepted_preds, ctx.papers, ctx.topic, ctx.domain,
            adversarial_results=_phases.get("adversarial_test") or None,
        )
        ctx.report["phases"]["peer_review"] = peer_review
        print(f"  Overall score: {peer_review.get('overall_score', 0)}/10")
        print(f"  Recommendation: {peer_review.get('recommendation', 'unknown')}")
        major = peer_review.get("major_issues", [])
        if major:
            print(f"  Major issues: {len(major)}")
            for issue in major[:2]:
                print(f"    ! {str(issue)[:80]}")


def _phase_theory_revision(ctx):
    print("\n[Phase 8.7/16] THEORY REVISION — Revising theories from review feedback...", flush=True)
    try:
        from discovery.theory_revision import TheoryRevisionEngine
        revision_engine = TheoryRevisionEngine(llm_call=_truncated_llm)
        rev_result = revision_engine.revise_theories(
            theories=ctx.theories, winner=ctx.winner or {},
            adversarial_results=ctx.report["phases"].get("adversarial_test", {}),
            critical_eval=ctx.report["phases"].get("peer_review", {}),
            mechanisms=ctx.mechanisms, predictions=ctx.accepted_preds,
            domain=ctx.domain, topic=ctx.topic, papers=ctx.papers,
            revision_round=1,
        )
        revised_theories = rev_result.get("revised_theories", [])
        if revised_theories:
            ctx.theories.extend(revised_theories)
            ctx.report["phases"]["theory_competition"]["theories"] = ctx.theories
            ctx.report["phases"]["theory_competition"]["revised_theories"] = revised_theories
        rev_summary = rev_result.get("summary", {})
        print(f"  Targeted: {rev_summary.get('targeted', 0)}, Revised: {rev_summary.get('revised', 0)}, Skipped: {rev_summary.get('skipped', 0)}")
        if rev_result.get("winner_replaced"):
            print(f"  Winner was replaced by revised theory")
        ctx.report["phases"]["theory_revision"] = rev_result
    except Exception as e:
        print(f"  [WARN] Theory revision skipped: {e}")


def _phase_computational_verification(ctx):
    print("\n[Phase 9/16] COMPUTATIONAL VERIFICATION — Running computations...", flush=True)
    verifier = ComputationalVerifier(graph=ctx.graph)
    top_theory = ctx.winner or (ctx.theories[0] if ctx.theories else {})
    verify_results = verifier.verify(top_theory, ctx.accepted_preds, ctx.mechanisms)
    comp_summary = verify_results.get("verification_summary", {})
    print(f"  Computations run: {verify_results.get('computations_run', 0)}")
    print(f"  Support level: {comp_summary.get('support_level', 'unknown')}")
    for finding in comp_summary.get("key_findings", [])[:3]:
        print(f"    → {finding[:80]}")
    ctx.report["phases"]["computational_verification"] = comp_summary


def _phase_math_verification(ctx):
    print("\n[Phase 9.3/16] ENHANCED MATH VERIFICATION — 6-category analysis...", flush=True)
    from discovery.math_engine import run_math_verification
    top_theory = ctx.winner or (ctx.theories[0] if ctx.theories else {})
    if top_theory:
        math_verify = run_math_verification(top_theory, mechanisms=ctx.mechanisms, predictions=ctx.accepted_preds, llm_call=_truncated_llm)
        math_score = math_verify.get("overall_score", 0)
        math_summary = math_verify.get("summary", "")
        print(f"  Math score: {math_score}/100")
        print(f"  Summary: {math_summary[:100]}")
        ctx.report["phases"]["enhanced_math_verification"] = {
            "overall_score": math_score,
            "summary": math_summary,
            "equation_solving": math_verify.get("equation_solving", {}).get("equations_found", 0),
            "dimensional_analysis": math_verify.get("dimensional_analysis", {}).get("consistent", 0),
            "kinetic_validation": math_verify.get("kinetic_validation", {}).get("equations_found", 0),
        }


def _phase_computation_engine(ctx):
    print("\n[Phase 9.4/16] COMPUTATION ENGINE — Verifying mechanism numbers...", flush=True)
    from discovery.computation_engine import verify_mechanism_numbers, format_verification_report
    top_mechanisms = ctx.mechanisms[:5] if ctx.mechanisms else []
    verification_results = []
    for mech in top_mechanisms:
        v = verify_mechanism_numbers(mech)
        verification_results.append(v)
    n_verified = sum(len(v.get("verified_params", [])) for v in verification_results)
    n_unverifiable = sum(len(v.get("unverifiable_params", [])) for v in verification_results)
    n_derived = sum(len(v.get("derivation_chains", [])) for v in verification_results)
    print(f"  Verified: {n_verified}, Unverifiable: {n_unverifiable}, Derivation chains: {n_derived}")
    ctx.report["phases"]["computation_engine"] = {
        "mechanisms_checked": len(top_mechanisms),
        "verified_params": n_verified,
        "unverifiable_params": n_unverifiable,
        "derivation_chains": n_derived,
    }


def _phase_derivation_engine(ctx):
    print("\n[Phase 9.45/16] DERIVATION ENGINE — Verifying mechanism equations...", flush=True)
    from discovery.derivation_engine import run_domain_verification
    derivation_results = run_domain_verification(ctx.domain, ctx.mechanisms, ctx.topic)
    source = derivation_results.get("source", "unknown")

    # Print mechanism-driven results
    if "mechanisms_verified" in derivation_results:
        n_verified = derivation_results.get("mechanisms_verified", 0)
        n_eq_found = derivation_results.get("equations_found", 0)
        n_eq_ok = derivation_results.get("equations_verified", 0)
        n_eq_fail = derivation_results.get("equations_failed", 0)
        print(f"  Source: {source}")
        print(f"  Mechanisms with parseable equations: {n_verified}/{len(ctx.mechanisms or [])}")
        print(f"  Equations found: {n_eq_found}, verified: {n_eq_ok}, failed: {n_eq_fail}")

        # Print per-mechanism details
        for md in derivation_results.get("mechanism_details", [])[:5]:
            n_eqs = len(md.get("equations", []))
            n_parsed = sum(1 for e in md.get("equations", []) if e.get("symbolically_parsed") or e.get("numerically_evaluated"))
            verdict = md.get("verdict", "?")
            print(f"    [{verdict}] {md.get('name', '?')}: {n_parsed}/{n_eqs} equations verified")
            # Show numeric results if available
            for eq in md.get("equations", []):
                if eq.get("numerical_value") is not None:
                    print(f"      {eq.get('equation', '?')[:60]} = {eq['numerical_value']}")
    else:
        # Fallback: domain-specific results (Casimir, BCS, etc.)
        print(f"  Source: {source} (domain template fallback)")
        multi_sep = derivation_results.get("multi_sep", [])
        scaling = derivation_results.get("scaling_100_200nm", {})
        gold = derivation_results.get("gold_100nm", {})
        uncert = derivation_results.get("uncert_1um_1nm", {})
        if "bcs_gap" in derivation_results:
            bcs = derivation_results.get("bcs_gap", {})
            print(f"  BCS gap: {bcs.get('gap_Delta_0_meV', 0):.2f} meV, T_c BCS: {bcs.get('T_c_BCS_K', 0):.1f} K")
        elif "tidal_heating" in derivation_results:
            tidal = derivation_results.get("tidal_heating", {})
            print(f"  Tidal heating: {tidal.get('power_GW', 0):.3f} GW")
        elif multi_sep:
            print(f"  Computed {len(multi_sep)} separations")

    # Store full results
    derivation_summary = {"details": derivation_results, "source": source}
    ctx.report["phases"]["derivation_engine"] = derivation_summary


def _phase_scientific_simulator(ctx):
    print("\n[Phase 9.46/16] SCIENTIFIC SIMULATOR — Running computations...", flush=True)
    sim = ScientificSimulator()
    sim_results = sim.simulate(
        mechanisms=ctx.mechanisms[:5] if ctx.mechanisms else [],
        hidden_variables=ctx.hidden_variables[:5] if ctx.hidden_variables else [],
        predictions=ctx.accepted_preds[:5] if ctx.accepted_preds else [],
        topic=ctx.topic,
        domain=ctx.domain,
    )
    n_equations = len(sim_results.get("equation_results", []))
    n_sweeps = len(sim_results.get("parameter_sweeps", []))
    n_observables = len(sim_results.get("observables", []))
    n_consistency = len(sim_results.get("consistency_checks", []))
    tools = sim_results.get("tools_used", [])
    print(f"  Equations solved: {n_equations}")
    print(f"  Parameter sweeps: {n_sweeps}")
    print(f"  Observables computed: {n_observables}")
    print(f"  Consistency checks: {n_consistency}")
    print(f"  Tools used: {tools}")
    ctx.report["phases"]["scientific_simulator"] = {
        "equations_solved": n_equations,
        "parameter_sweeps": n_sweeps,
        "observables_computed": n_observables,
        "consistency_checks": n_consistency,
        "tools_used": tools,
        "summary": sim_results.get("summary", {}),
    }


_CASIMIR_RE = _re_module.compile(r"casimir|van\s*der\s*waals|\bgold\b", _re_module.IGNORECASE)


def _casimir_relevance_text(ctx) -> str:
    text_parts = [ctx.topic]
    for m in (ctx.mechanisms or []):
        if isinstance(m, dict):
            text_parts.append(str(m.get("name", "")))
            text_parts.append(str(m.get("description", "")))
            text_parts.append(str(m.get("mathematical_model", "")))
    return " ".join(text_parts)


def _phase_uncertainty_analysis(ctx):
    print("\n[Phase 9.47/16] UNCERTAINTY ANALYSIS — Error propagation...", flush=True)
    from discovery.derivation_engine import propagate_uncertainty, gold_correction
    uncertainty_results = {}

    if _CASIMIR_RE.search(_casimir_relevance_text(ctx)):
        # Casimir/gold reference computations — only for actually-Casimir runs.
        # Standard Casimir uncertainty at key separations
        for a_nm, da_nm in [(50, 0.5), (100, 0.1), (200, 0.2), (500, 0.5), (1000, 1.0)]:
            key = f"uncert_{a_nm}nm"
            uncertainty_results[key] = propagate_uncertainty(a_nm * 1e-9, da_nm * 1e-9)

        # Gold correction at key separations
        for a_nm in [50, 100, 200, 500, 1000, 5000]:
            key = f"gold_{a_nm}nm"
            uncertainty_results[key] = gold_correction(a_nm * 1e-9)

        # Print summary
        for key in sorted(uncertainty_results.keys()):
            r = uncertainty_results[key]
            if "rel_pct" in r:
                print(f"    {key}: relative uncertainty = {r['rel_pct']}%")
            elif "deviation_pct" in r:
                print(f"    {key}: gold deviation = {r['deviation_pct']}%")
    else:
        # GENERIC path: propagate the mechanisms' own declared uncertainties.
        # The old code ran Casimir/gold plate computations on every topic —
        # contaminating exoplanet/climate/ISRO reports with irrelevant physics.
        n_propagated = 0
        for m in (ctx.mechanisms or [])[:5]:
            if not isinstance(m, dict):
                continue
            for kp in (m.get("key_parameters") or []):
                if not isinstance(kp, dict):
                    continue
                value = kp.get("value", kp.get("expected_value"))
                uncertainty = kp.get("uncertainty")
                try:
                    value = float(str(value))
                    uncertainty = float(str(uncertainty))
                except (TypeError, ValueError):
                    continue
                if value == 0:
                    continue
                rel_pct = round(abs(uncertainty / value) * 100, 2)
                key = f"parameter_{kp.get('name', '?')}"
                uncertainty_results[key] = {
                    "mechanism": m.get("name", "?"),
                    "parameter": kp.get("name", "?"),
                    "value": value,
                    "uncertainty": uncertainty,
                    "units": kp.get("units", ""),
                    "rel_pct": rel_pct,
                }
                n_propagated += 1
                print(f"    {kp.get('name', '?')}: relative uncertainty = {rel_pct}% "
                      f"({value} ± {uncertainty} {kp.get('units', '')})")
        if n_propagated == 0:
            print("  Casimir/gold template skipped (not a Casimir topic) and no mechanism "
                  "parameters declare numeric uncertainties — nothing to propagate")

    ctx.report["phases"]["uncertainty_analysis"] = uncertainty_results


def _phase_experiment_planning(ctx):
    print("\n[Phase 9.5/16] EXPERIMENTAL VALIDATION — Designing experiments...", flush=True)
    CURRENT_PHASE["name"] = "critical_evaluation"  # use nvidia for experiment design
    from discovery.experiment_planner import ExperimentPlanner
    planner = ExperimentPlanner(llm_call=_truncated_llm)
    top_theory = ctx.winner or (ctx.theories[0] if ctx.theories else {})
    if top_theory:
        exp_plans = planner.plan_for_top_theories(
            (ctx.theories or [])[:3], ctx.mechanisms or [], ctx.accepted_preds or [], ctx.papers, ctx.topic, ctx.domain, max_plans=2
        )
        if exp_plans:
            print(f"  Generated {len(exp_plans)} experimental validation plans")
            for ep in exp_plans:
                etype = ep.get("experiment_type", "?")
                tname = ep.get("theory_name", "?")
                cost = ep.get("estimated_cost", "?")
                timeline = ep.get("timeline_estimate", "?")
                print(f"    [{etype}] {tname[:40]} — {timeline}, {cost} cost")
            ctx.report["phases"]["experimental_validation"] = {
                "plans_generated": len(exp_plans),
                "plans": exp_plans,
            }
        else:
            print("  No experimental plans generated")


# Public registry hook so extensible data analysis is config-driven, not
# keyword-matched. Add a source = one entry here + one `data_sources` list
# entry in discovery.domains.
DATASET_RUNNERS = {
    "nasa_exoplanets": ("fetch_nasa_exoplanets", {"limit": 50}),
    "nasa_neo": ("fetch_nasa_neo", {"days": 7}),
}


def _phase_data_analysis(ctx):
    print("\n[Phase 9.7/16] DATA ANALYSIS — Fetching public datasets...", flush=True)
    from discovery.data_analysis import DataAnalyzer
    from discovery.domains import get_domain
    analyzer = DataAnalyzer()
    dataset_results = {}

    # Auto-select data source from the DOMAIN CONFIG (domains.py `data_sources`),
    # not from substring-matching the domain name.
    domain_cfg = get_domain(ctx.domain) or {}
    data_sources = domain_cfg.get("data_sources") or []
    for source in data_sources:
        runner = DATASET_RUNNERS.get(source)
        if runner is None:
            print(f"  Unknown data source '{source}' — no runner in DATASET_RUNNERS")
            continue
        method_name, kwargs = runner
        try:
            result = getattr(analyzer, method_name)(**kwargs)
        except Exception as fetch_err:
            print(f"  [WARN] {source} fetch failed: {fetch_err}")
            continue
        if result.get("status") == "ok":
            dataset_results[source] = result
            analysis = analyzer.analyze_dataset(source)
            dataset_results["analysis"] = analysis
            print(f"  {source}: {result['records']} records, {len(analysis.get('statistics', {}))} numeric fields")
            if analysis.get("correlations"):
                top_corr = analysis["correlations"][0]
                print(f"    Top correlation: {top_corr['field1']} ↔ {top_corr['field2']} (r={top_corr['correlation']})")

    if dataset_results:
        ctx.report["phases"]["data_analysis"] = {
            "datasets_fetched": len(dataset_results),
            "results": {k: {"status": v.get("status"), "records": v.get("records", 0)} for k, v in dataset_results.items() if isinstance(v, dict)},
            "statistics": dataset_results.get("analysis", {}).get("statistics", {}),
            "correlations": dataset_results.get("analysis", {}).get("correlations", []),
        }
    else:
        print("  No domain-specific datasets available")


def _phase_contradiction_mining(ctx):
    print("\n[Phase 10/16] CONTRADICTION MINING...", flush=True)
    miner = ContradictionMiner(ctx.graph)
    contradiction_result = miner.mine()
    contradictions = contradiction_result.get("contradictions", [])
    graph_contradictions = ctx.graph.detect_contradictions()
    all_contradictions = contradictions + [
        {"type": c["type"], "description": c["summary"], "severity": c.get("severity", "medium")}
        for c in graph_contradictions
    ]
    print(f"  Found {len(all_contradictions)} contradictions")
    ctx.report["phases"]["contradictions"] = {
        "total": len(all_contradictions),
        "contradictions": all_contradictions,
    }
    ctx.all_contradictions = all_contradictions


def _phase_skeptic_review(ctx):
    print("\n[Phase 11/16] SKEPTIC REVIEW — Adversarial critique...", flush=True)
    CURRENT_PHASE["name"] = "skeptic_review"
    skeptic_result = {}
    top_theory = ctx.winner or (ctx.theories[0] if ctx.theories else {})
    if top_theory:
        # Build context about competing theories for fair comparison
        alt_names = [t.get('name', '?') for t in ctx.theories[:5] if t.get('name') != top_theory.get('name')]
        alt_text = ', '.join(alt_names[:3]) if alt_names else 'none'

        skeptic_prompt = f"""You are a rigorous but fair scientific reviewer. Evaluate this theory objectively.

THEORY: {top_theory.get('name', '?')}
DESCRIPTION: {top_theory.get('description', top_theory.get('mechanism', ''))[:300]}
PREDICTIONS: {json.dumps(top_theory.get('predictions', [])[:3])}
COMPETING THEORIES: {alt_text}
PAPERS SUPPORTING: {len(ctx.papers)}

Evaluate FAIRLY:
1. What does this theory explain well? (strengths)
2. What are the genuine weaknesses? (be specific)
3. How does it compare to alternatives?
4. What evidence would strengthen or weaken it?
5. Recommendation: accept (strong) | revise (promising but needs work) | reject (fundamentally flawed)
6. Revised confidence (0.0-1.0) — be realistic, not automatically low
7. FAILURE CONDITIONS: What specific observation would definitively disprove this theory?
8. DESTROYING EVIDENCE: What existing evidence contradicts or weakens this theory?
9. COMPETING EXPLANATION: What alternative theory explains the same data better?

IMPORTANT: Only recommend "reject" if the theory has FUNDAMENTAL logical flaws or contradicts established evidence. Most theories should get "revise" or "accept with conditions".

Output JSON: {{"critique": "...", "strengths": ["s1", "s2"], "weaknesses": ["w1", "w2"], "recommendation": "accept|revise|reject", "revised_confidence": 0.0, "improvement_suggestions": ["suggestion1"], "failure_conditions": ["condition1"], "destroying_evidence": ["evidence1"], "competing_explanation": "alternative theory"}}"""

        raw = _truncated_llm(skeptic_prompt, max_tokens=4096)
        if raw:
            from discovery.json_extract import extract_json
            skeptic_result = extract_json(raw)
            if skeptic_result is None:
                # Fallback: try regex extraction
                if isinstance(raw, str):
                    match = _re_module.search(r'\{.*\}', raw, _re_module.DOTALL)
                    if match:
                        try:
                            skeptic_result = json.loads(match.group())
                        except json.JSONDecodeError:
                            pass

    rec = skeptic_result.get('recommendation', 'unknown')
    conf = skeptic_result.get('revised_confidence', 0)
    strengths = skeptic_result.get('strengths', [])
    weaknesses = skeptic_result.get('weaknesses', [])
    print(f"  Recommendation: {rec} (confidence: {conf:.0%})")
    if strengths:
        print(f"  Strengths: {', '.join(strengths[:2])}")
    if weaknesses:
        print(f"  Weaknesses: {', '.join(weaknesses[:2])}")
    ctx.report["phases"]["skeptic_review"] = {
        "recommendation": rec,
        "revised_confidence": conf,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }
    ctx.skeptic_result = skeptic_result


def _phase_post_skeptic_revision(ctx):
    print("\n[Phase 11.5/16] POST-SKEPTIC REVISION — Final polish...", flush=True)
    try:
        from discovery.theory_revision import TheoryRevisionEngine
        revision_engine = TheoryRevisionEngine(llm_call=_truncated_llm)
        post_skeptic = revision_engine.post_skeptic_revision(
            winner=ctx.winner or {}, skeptic_result=ctx.skeptic_result,
            domain=ctx.domain, topic=ctx.topic,
        )
        if post_skeptic.get("winner_updated"):
            revised = post_skeptic.get("revised_theory", {})
            if revised:
                ctx.winner = revised
                # Update winner in theories list
                for i, t in enumerate(ctx.theories):
                    if t.get("name") == ctx.winner.get("_revision_source"):
                        ctx.theories[i] = ctx.winner
                        break
                ctx.report["phases"]["theory_competition"]["winner"] = ctx.winner
                ctx.report["phases"]["theory_competition"]["theories"] = ctx.theories
                print(f"  Winner updated: {ctx.winner.get('name', '?')}")
        ctx.report["phases"]["post_skeptic_revision"] = post_skeptic
    except Exception as e:
        print(f"  [WARN] Post-skeptic revision skipped: {e}")


def _phase_discovery_scoring(ctx):
    print("\n[Phase 12/16] DISCOVERY SCORING — Final quality gate...", flush=True)
    CURRENT_PHASE["name"] = "scoring"
    scorer = DiscoveryScorer()
    top_theory = ctx.winner or (ctx.theories[0] if ctx.theories else {})
    if top_theory:
        # Get adversarial and peer review results for feedback
        adv_results = ctx.report["phases"].get("adversarial_test", {})
        peer_rev = ctx.report["phases"].get("peer_review", {})
        score_result = scorer.score(
            top_theory, ctx.gaps, ctx.anomalies, ctx.accepted_preds, ctx.papers, ctx.graph,
            adversarial_results=adv_results if adv_results else None,
            peer_review=peer_rev if peer_rev else None,
        )
        # Apply domain-specific scoring weights
        try:
            from discovery.domain_templates import adjust_discovery_score
            score_result = adjust_discovery_score(score_result, ctx.domain)
            domain_score = score_result.get("domain_weighted_score", score_result["discovery_score"])
            print(f"  Discovery Score: {score_result['discovery_score']:.0f}/100 (domain-weighted: {domain_score:.0f}/100)")
        except Exception:
            print(f"  Discovery Score: {score_result['discovery_score']:.0f}/100")
        print(f"  Grade: {score_result['grade']}")
        print(f"  Summary: {score_result['summary'][:100]}")
        ctx.report["phases"]["discovery_scoring"] = score_result
    else:
        ctx.report["phases"]["discovery_scoring"] = {"discovery_score": 0, "grade": "F"}


def _phase_cross_validation(ctx):
    print("\n[Phase 12.5/16] CROSS-VALIDATION — Testing against held-out papers...", flush=True)
    from discovery.cross_validation import CrossValidator
    cv = CrossValidator()
    top_theory = ctx.winner or (ctx.theories[0] if ctx.theories else {})
    if top_theory and ctx.papers:
        cv_result = cv.validate(top_theory, ctx.papers, ctx.topic, ctx.domain)
        robustness = cv_result.get("robustness_score", 0.5)
        print(f"  Holdout papers: {cv_result.get('holdout_papers', 0)}")
        print(f"  Supported: {cv_result.get('supported_by_holdout', 0)}, "
              f"Contradicted: {cv_result.get('contradicted_by_holdout', 0)}, "
              f"Neutral: {cv_result.get('neutral_by_holdout', 0)}")
        print(f"  Robustness score: {robustness:.3f}")

        # Feed robustness into discovery score (constants in scoring_config)
        ds = ctx.report["phases"].get("discovery_scoring", {})
        if ds and ds.get("discovery_score", 0) > 0:
            orig_score = ds["discovery_score"]
            adjustment = cross_validation_adjustment(robustness)
            if adjustment["kind"] == "penalty":
                penalty = adjustment["amount"]
                adjusted = max(0, orig_score - penalty)
                ds["discovery_score"] = round(adjusted, 1)
                ds["cross_validation_penalty"] = penalty
                print(f"  Score adjusted: {orig_score:.0f} → {adjusted:.0f} (robustness penalty: {penalty:.1f})")
            elif adjustment["kind"] == "bonus":
                ds["cross_validation_bonus"] = adjustment["amount"]
                ds["discovery_score"] = round(min(100, orig_score + adjustment["amount"]), 1)
                print(f"  Score adjusted: {orig_score:.0f} → {ds['discovery_score']:.0f} (robustness bonus)")
            else:
                print(f"  Score unchanged (robustness exactly at threshold)")
            # Update grade from the shared threshold table
            ds["grade"] = grade_for(ds["discovery_score"])

        ctx.report["phases"]["cross_validation"] = cv_result
    else:
        print(f"  Skipped: not enough papers or no theory")


def _phase_classification(ctx):
    """Classify discovery type — check ACTUAL mechanism content."""
    top_theory = ctx.winner or (ctx.theories[0] if ctx.theories else {})
    theory_desc = top_theory.get("description", top_theory.get("mechanism", "")) if top_theory else ""

    # Check mechanisms for real content
    has_equations = False
    has_predictions = False
    has_novel_mechanism = False

    for m in (ctx.mechanisms or []):
        if not isinstance(m, dict):
            continue
        model = m.get("mathematical_model", "")
        if model and ("=" in model or "pi" in model.lower() or "hbar" in model.lower()):
            has_equations = True
        if m.get("predictions"):
            has_predictions = True
        novel = m.get("is_novel_vs_known", "")
        if novel in ("novel", "extension_of_known"):
            has_novel_mechanism = True

    # Check theories for novelty
    for t in (ctx.theories or []):
        if isinstance(t, dict):
            novel = t.get("is_novel_vs_known", t.get("novelty", ""))
            if novel in ("novel", "high"):
                has_novel_mechanism = True
            if t.get("mathematical_model"):
                has_equations = True

    has_new_prediction = len(ctx.accepted_preds) > 0
    not_in_literature = top_theory.get("is_novel_vs_known", "") == "novel" if top_theory else False

    if has_novel_mechanism and has_new_prediction and has_equations and not_in_literature:
        discovery_class = "novel_theory"
    elif has_novel_mechanism or has_new_prediction:
        discovery_class = "extension"
    elif len(ctx.theories) > 1:
        discovery_class = "synthesis"
    else:
        discovery_class = "replication"

    ctx.report["phases"]["discovery_classification"] = {
        "classification": discovery_class,
        "novel_mechanism": has_novel_mechanism,
        "novel_prediction": has_new_prediction,
        "novel_mathematics": has_equations,
        "not_in_literature": not_in_literature,
        "mechanisms_with_equations": sum(1 for m in (ctx.mechanisms or []) if isinstance(m, dict) and m.get("mathematical_model")),
        "theories_compared": len(ctx.theories or []),
    }
    print(f"  Classification: {discovery_class}")
    print(f"    Novel mechanism: {has_novel_mechanism}")
    print(f"    Novel prediction: {has_new_prediction}")
    print(f"    Equations: {has_equations}")


def _phase_refinement(ctx):
    print("\n" + "=" * 70, flush=True)
    print("  REFINEMENT PIPELINE — 13 stages of post-processing", flush=True)
    print("=" * 70, flush=True)

    from discovery.refinement_pipeline import run_refinement_pipeline
    refinement = run_refinement_pipeline(
        ctx.topic, ctx.domain, ctx.papers, ctx.graph,
        ctx.theories, ctx.all_contradictions
    )
    ctx.report["refinement"] = refinement

    # Print refinement summary
    if refinement:
        print(f"\n  Stage 1 (Knowledge Audit): {len(refinement.get('audit', {}).get('key_entities', refinement.get('audit', {}).get('known_facts', [])))} items")
        print(f"  Stage 5 (Multi-Model): {len(refinement.get('multi_model', refinement.get('competition', {})).get('hypotheses', []))} hypotheses")
        classification = refinement.get('classification', {})
        print(f"  Stage 11 (Classification): {classification.get('classification', 'unknown')}")
        scoring = refinement.get('scoring', {})
        # Try both field names for compatibility
        score_val = scoring.get('overall_score', scoring.get('total_score', 0))
        print(f"  Stage 12 (Scoring): {scoring.get('grade', '?')} ({score_val:.0f}/100)")
        courtroom = refinement.get('courtroom', {})
        print(f"  Stage 13 (Courtroom): {courtroom.get('verdict', 'unknown')}")
        ctx.report["phases"]["refinement"] = {
            "classification": classification.get('classification', 'unknown'),
            "grade": scoring.get('grade', '?'),
            "score": score_val,
            "verdict": courtroom.get('verdict', 'unknown'),
            "scores": scoring.get('scores', {}),
        }

        # ── SCORE UNIFICATION: refinement score updates final grade ──
        # If refinement is harsher, use its score as the final grade.
        # Blend/cap/override constants live in scoring_config.
        discovery_scoring = ctx.report.get("phases", {}).get("discovery_scoring", {})
        if discovery_scoring and score_val > 0:
            original_score = discovery_scoring.get("discovery_score", 0)
            unification = unify_scores(original_score, score_val)
            unified_score = unification["unified_score"]
            if unification["capped"]:
                print(f"  [Score Unification] Refinement much lower ({score_val:.0f} vs {original_score:.0f}) — capping unified score")

            unified_grade = grade_for(unified_score)

            # Add refinement weaknesses to discovery weaknesses
            ref_weaknesses = discovery_scoring.get("weaknesses", [])
            if score_val < UNIFIED_WEAKNESS_REFINEMENT_SCORE:
                ref_weaknesses.append(f"refinement_grade: {scoring.get('grade', '?')} ({score_val:.0f}/100)")
            # Add courtroom verdict
            if courtroom.get("verdict") and courtroom.get("verdict") != "unknown":
                survived = courtroom.get("survived", True)
                if not survived:
                    ref_weaknesses.append(f"courtroom: {courtroom.get('verdict')} (survived: {survived})")

            # Store unified score
            ctx.report["phases"]["discovery_scoring"]["unified_score"] = unified_score
            ctx.report["phases"]["discovery_scoring"]["unified_grade"] = unified_grade
            ctx.report["phases"]["discovery_scoring"]["refinement_score"] = score_val
            ctx.report["phases"]["discovery_scoring"]["weaknesses"] = ref_weaknesses

            # If unified score is significantly lower, override discovery score
            if unification["override_discovery"]:
                ctx.report["phases"]["discovery_scoring"]["discovery_score"] = unified_score
                ctx.report["phases"]["discovery_scoring"]["grade"] = unified_grade
                print(f"  [Score Unification] Final: {unified_score:.0f}/100 ({unified_grade}) — refinement pulled score down")
            else:
                print(f"  [Score Unification] Final: {original_score:.0f}/100 — refinement consistent")


def _phase_reflexion(ctx):
    print("\n" + "=" * 70, flush=True)
    print("  REFLEXION — Recursive Self-Improvement", flush=True)
    print("=" * 70, flush=True)

    from brain.reflexion import get_recursive_improver
    improver = get_recursive_improver()
    run_result = {
        "query": ctx.topic,
        "domain": ctx.domain,
        "hypotheses": ctx.theories,
        "contradictions": ctx.all_contradictions,
        "metrics": {},
        "errors": ctx.report.get("errors", []),
    }
    reflexion_result = improver.reflect_and_improve_sync(run_result)
    ctx.report["reflexion"] = reflexion_result
    print(f"  Weaknesses found: {reflexion_result.get('weaknesses_found', 0)}")
    print(f"  Patches applied: {reflexion_result.get('patches_applied', 0)}")
    print(f"  Patches rejected: {reflexion_result.get('patches_rejected', 0)}")
    print(f"  Improvement score: {reflexion_result.get('improvement_score', 0):.0%}")


def _phase_self_improvement(ctx):
    print("\n[Phase 15/16] SELF-IMPROVEMENT ANALYSIS...", flush=True)
    CURRENT_PHASE["name"] = "self_improvement"
    from brain.self_modifier import get_self_modifier
    sm = get_self_modifier()
    analysis = sm.analyze_discovery_run(ctx.report)
    ctx.report["phases"]["self_improvement"] = analysis
    print(f"  Analysis complete. Performance score: {analysis.get('performance_score', 0):.2f}")


def _phase_gap_discovery(ctx):
    print("\n[Phase 16/16] KNOWLEDGE GAP AUTO-DISCOVERY...", flush=True)
    CURRENT_PHASE["name"] = "gap_discovery"
    kg_detector = KnowledgeGapDetector(graph=ctx.graph, llm_call=_truncated_llm)
    future_gaps = kg_detector.identify_unanswered_questions(ctx.report)
    ctx.report["phases"]["gap_discovery"] = {"next_questions": future_gaps}
    if future_gaps:
        print(f"  Identified {len(future_gaps)} priority research questions for next run.")


# ══════════════════════════════════════════════════════════════════
# THE REGISTRY — execution order + declarative mode gating.
# min_mode: lowest mode that includes this phase.
# optional: True → phase can be skipped by `enabled` (or failure) without
#           being reported as a pipeline error; False → failure is recorded.
# ══════════════════════════════════════════════════════════════════

class PhaseStep:
    __slots__ = ("num", "key", "label", "min_mode", "fn", "enabled", "critical")

    def __init__(self, num, key, label, min_mode, fn, enabled=None, critical=False):
        self.num = num
        self.key = key
        self.label = label
        self.min_mode = min_mode
        self.fn = fn
        self.enabled = enabled            # None = always run (within its mode)
        self.critical = critical          # True → failure recorded as error


def _enable_citation_network(ctx):
    return bool(ctx.papers)


def _enable_literature_refinement(ctx):
    return bool(ctx.gaps) and len(ctx.papers) < 50


def _enable_prediction_stages(ctx):
    return bool(ctx.accepted_preds)


PHASE_REGISTRY_ORDER = [
    # (num, key, label, min_mode, fn, enabled, critical)
    (1.0, "literature", "LITERATURE — Fetching papers from 4 sources (domain-aware)", "quick", _phase_literature, None, True),
    (1.5, "citation_network", "CITATION NETWORK — 2-hop citation walk", "quick", _phase_citation_network, _enable_citation_network, False),
    (2.0, "knowledge_graph", "KNOWLEDGE GRAPH — Building", "quick", _phase_knowledge_graph, None, True),
    (2.5, "enrichment", "ENRICHMENT — Calling external APIs", "quick", _phase_enrichment, None, False),
    (2.6, "graph_inference", "GRAPH INFERENCE — Reducing fragmentation", "quick", _phase_graph_inference, None, False),
    (3.0, "gap_detection", "GAP DETECTION — Finding what's missing", "quick", _phase_gap_detection, None, True),
    (3.5, "literature_refinement", "LITERATURE REFINEMENT — Targeted search based on gaps", "standard", _phase_literature_refinement, _enable_literature_refinement, False),
    (4.0, "anomaly_detection", "ANOMALY DETECTION — Finding what doesn't fit", "standard", _phase_anomaly_detection, None, True),
    (0.0, "curious_questioning", "CURIOUS QUESTIONING — The Newton Step", "standard", _phase_curious_questioning, None, False),
    (5.0, "missing_variables", "MISSING VARIABLES — Proposing hidden factors", "standard", _phase_missing_variables, None, True),
    (6.0, "mechanism_generation", "MECHANISM GENERATION — Building causal pathways", "full", _phase_mechanism_generation, None, True),
    (6.4, "derivation_validation", "DERIVATION VALIDATOR — Checking reproducibility", "full", _phase_derivation_validator, None, False),
    (6.5, "math_chain", "MATH CHAIN — Deriving equations for mechanisms", "full", _phase_math_chain, None, False),
    (7.0, "prediction_engine", "PREDICTION ENGINE — Generating testable predictions", "full", _phase_prediction_engine, None, True),
    (7.5, "observability", "OBSERVABILITY CHECK — Can we measure these predictions?", "full", _phase_observability, _enable_prediction_stages, False),
    (7.6, "prediction_literature_validation", "PREDICTION LITERATURE VALIDATION — Checking against published data", "full", _phase_prediction_literature_validation, _enable_prediction_stages, False),
    (8.0, "theory_competition", "THEORY COMPETITION — Comparing explanations", "full", _phase_theory_competition, None, True),
    (8.5, "adversarial_test", "ADVERSARIAL TEST — Attacking every discovery", "full", _phase_adversarial_test, None, True),
    (8.6, "peer_review", "CRITICAL EVALUATION — Formal assessment", "full", _phase_peer_review, None, False),
    (8.7, "theory_revision", "THEORY REVISION — Revising theories from review feedback", "full", _phase_theory_revision, None, False),
    (9.0, "computational_verification", "COMPUTATIONAL VERIFICATION — Running computations", "full", _phase_computational_verification, None, True),
    (9.3, "enhanced_math_verification", "ENHANCED MATH VERIFICATION — 6-category analysis", "full", _phase_math_verification, None, False),
    (9.4, "computation_engine", "COMPUTATION ENGINE — Verifying mechanism numbers", "full", _phase_computation_engine, None, False),
    (9.45, "derivation_engine", "DERIVATION ENGINE — Verifying mechanism equations", "full", _phase_derivation_engine, None, False),
    (9.46, "scientific_simulator", "SCIENTIFIC SIMULATOR — Running computations", "full", _phase_scientific_simulator, None, False),
    (9.47, "uncertainty_analysis", "UNCERTAINTY ANALYSIS — Error propagation", "full", _phase_uncertainty_analysis, None, False),
    (9.5, "experimental_validation", "EXPERIMENTAL VALIDATION — Designing experiments", "full", _phase_experiment_planning, None, False),
    (9.7, "data_analysis", "DATA ANALYSIS — Fetching public datasets", "full", _phase_data_analysis, None, False),
    (10.0, "contradiction_mining", "CONTRADICTION MINING", "full", _phase_contradiction_mining, None, True),
    (11.0, "skeptic_review", "SKEPTIC REVIEW — Adversarial critique", "full", _phase_skeptic_review, None, True),
    (11.5, "post_skeptic_revision", "POST-SKEPTIC REVISION — Final polish", "full", _phase_post_skeptic_revision, None, False),
    (12.0, "discovery_scoring", "DISCOVERY SCORING — Final quality gate", "full", _phase_discovery_scoring, None, True),
    (12.5, "cross_validation", "CROSS-VALIDATION — Testing against held-out papers", "full", _phase_cross_validation, None, False),
    (12.7, "discovery_classification", "DISCOVERY CLASSIFICATION — What's Known vs What's New", "full", _phase_classification, None, False),
    (13.0, "refinement", "REFINEMENT PIPELINE — 13 stages of post-processing", "full", _phase_refinement, None, False),
    (14.0, "reflexion", "REFLEXION — Recursive Self-Improvement", "full", _phase_reflexion, None, False),
    (15.0, "self_improvement", "SELF-IMPROVEMENT ANALYSIS", "full", _phase_self_improvement, None, False),
    (16.0, "gap_discovery", "KNOWLEDGE GAP AUTO-DISCOVERY", "full", _phase_gap_discovery, None, False),
]

PHASE_REGISTRY = [PhaseStep(*entry) for entry in PHASE_REGISTRY_ORDER]


def run_phase(step: PhaseStep, ctx: PipelineContext) -> None:
    """Execute one registry step with uniform error handling."""
    if step.critical:
        step.fn(ctx)   # outsourcing error handling to run_discovery_pipeline's own try
        return
    try:
        step.fn(ctx)
    except Exception as e:
        print(f"  [ERROR] Phase {step.num} ({step.key}) failed: {e}", flush=True)
        ctx.report["errors"].append(f"Phase {step.num}: {e}")


def list_phases(mode: str = None) -> list:
    """Introspection helper: [(num, key, min_mode), ...] for diagnostics/tests."""
    level = MODE_LEVELS.get(mode, 3) if mode else None
    out = []
    for step in PHASE_REGISTRY:
        if level is not None and MODE_LEVELS[step.min_mode] > level:
            continue
        out.append((step.num, step.key, step.min_mode))
    return out


def _run_phases(ctx: PipelineContext) -> None:
    """Execute every registry phase whose mode gate is satisfied."""
    mode_level = MODE_LEVELS.get(ctx.mode, 3)
    for step in PHASE_REGISTRY:
        if MODE_LEVELS[step.min_mode] > mode_level:
            continue
        if step.enabled is not None and not step.enabled(ctx):
            print(f"\n[Phase {step.num}/16] {step.label}... (skipped: prerequisites not met)", flush=True)
            continue
        if step.critical:
            # Critical phases keep their own targeted error reporting, matching
            # the original per-phase error messages.
            try:
                step.fn(ctx)
            except Exception as e:
                print(f"  [ERROR] {step.label.split(' — ')[0].title()} failed: {e}", flush=True)
                ctx.report["errors"].append(f"Phase {step.num}: {e}")
        else:
            run_phase(step, ctx)


def run_discovery_pipeline(topic: str, domain: str = "", mode: str = "full",
                           curiosity_constraint: dict = None,
                           original_topic: str = "") -> dict:
    """
    Run the complete discovery pipeline via the phase registry.

    Args:
        topic: Research topic
        domain: Domain (auto-detected if empty)
        mode: "quick" (foundation phases), "standard" (+ analysis), "full" (everything)

    Returns:
        Complete discovery report dict
    """
    if not domain:
        domain = detect_domain(topic)

    status = get_status()
    t_start = time.time()
    provider_name = status.get("primary", "auto") if isinstance(status, dict) else "auto"

    ctx = PipelineContext(topic, domain, mode, curiosity_constraint, original_topic, provider_name)
    report = ctx.report

    # Load discovery archive — what Luka already knows from past runs
    from discovery.discovery_archive import get_archive_context, save_to_archive
    archive_context = get_archive_context(topic, domain)
    if "No previous" not in archive_context:
        print(f"  [Archive] Loaded past discovery context", flush=True)
    ctx.archive_context = archive_context

    # Load hypothesis memory — prior hypotheses from all runs
    hyp_mem = HypothesisMemory()
    prior_hypotheses = hyp_mem.get_all(domain=domain, limit=20)
    # Also fetch by topic similarity
    topic_similar = hyp_mem.find_similar(topic, threshold=0.3)
    prior_hyp_ids = {h.get("id") for h in prior_hypotheses}
    for ts in topic_similar:
        if ts["id"] not in prior_hyp_ids:
            full = hyp_mem.get_hypothesis(ts["id"])
            if full:
                prior_hypotheses.append(full)
                prior_hyp_ids.add(ts["id"])
    ctx.hyp_mem = hyp_mem

    # Build prior hypothesis context and merge into archive_context
    if prior_hypotheses:
        prior_lines = []
        prior_lines.append(f"\nPRIOR HYPOTHESIS MEMORY ({len(prior_hypotheses)} hypotheses from past runs):")
        prior_lines.append("These were proposed before. Build on them, refine them, or explore angles they missed.")
        for h in prior_hypotheses[:10]:
            status_h = h.get("status", "draft")
            title = h.get("title", "?")
            desc = (h.get("description") or "")[:150]
            score = h.get("confidence", 0)
            prior_lines.append(f"  [{status_h}] {title} (confidence: {score:.2f})")
            if desc:
                prior_lines.append(f"    {desc}")
            weaknesses = h.get("critique_weaknesses", "")
            if weaknesses:
                prior_lines.append(f"    Weaknesses: {weaknesses[:100]}")
        archive_context += "\n" + "\n".join(prior_lines)
        print(f"  [Hypothesis Memory] Loaded {len(prior_hypotheses)} prior hypotheses", flush=True)

    # Add domain-specific research template context
    try:
        from discovery.domain_templates import get_research_question_prompt, get_validation_prompt
        domain_q = get_research_question_prompt(domain, topic)
        domain_v = get_validation_prompt(domain)
        if domain_q:
            archive_context += "\n\n" + domain_q
        if domain_v:
            archive_context += "\n" + domain_v
        print(f"  [Domain Template] Loaded {domain} research context", flush=True)
    except Exception:
        pass
    ctx.archive_context = archive_context

    print("=" * 70, flush=True)
    print("  Luka — DISCOVERY PIPELINE v2 (phase registry)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Topic:     {topic}")
    print(f"  Domain:    {domain}")
    print(f"  Mode:      {mode}")
    print(f"  Provider:  {provider_name.upper()}")
    print(f"  Phases:    {len(list_phases(mode))} enabled at this mode")
    print("=" * 70)

    mode_level = MODE_LEVELS.get(mode, 3)

    # ── RUN ALL PHASES ──
    _run_phases(ctx)

    # ── FINALIZE (full-mode housekeeping: archive + hypothesis memory) ──
    if mode_level >= 3:
        try:
            run_summary = save_to_archive(report, topic, domain)
            print(f"  [Archive] Saved: {run_summary.get('theories_count', 0)} theories, "
                  f"{run_summary.get('variables_count', 0)} variables, "
                  f"{run_summary.get('mechanisms_count', 0)} mechanisms", flush=True)
        except Exception as e:
            print(f"  [Archive] Save failed: {e}", flush=True)

        # Save theories to hypothesis memory — cross-run persistence
        try:
            saved_count = 0
            for t in ctx.theories:
                if isinstance(t, dict) and t.get("name"):
                    hyp_data = {
                        "title": t.get("name", ""),
                        "description": t.get("description", t.get("mechanism", "")),
                        "pattern_type": t.get("type", "proposed"),
                        "confidence": t.get("scores", {}).get("overall", 0),
                        "novelty": t.get("is_novel_vs_known", ""),
                        "status": "accepted" if ctx.winner and t.get("name") == ctx.winner.get("name") else "survived",
                        "topic": topic,
                        "domain": domain,
                        "testability": json.dumps(t.get("predictions", [])[:3]),
                        "nodes": t.get("hidden_variables", []),
                        "edges": [],
                    }
                    hyp_mem.save_hypothesis(hyp_data, run_id=ctx.run_id)
                    saved_count += 1
            if saved_count:
                print(f"  [Hypothesis Memory] Saved {saved_count} theories", flush=True)
            report["hypothesis_memory"] = {"saved": saved_count, "run_id": ctx.run_id}
        except Exception as e:
            print(f"  [Hypothesis Memory] Save failed: {e}", flush=True)

    # ── ASSEMBLE FINAL REPORT ──
    return _finalize_report(report, ctx.papers, ctx.graph, ctx.gaps, ctx.anomalies,
                            ctx.hidden_variables, ctx.mechanisms, ctx.predictions,
                            ctx.theories, ctx.accepted_preds, ctx.t_start)

def _finalize_report(report, papers, graph, gaps, anomalies,
                     hidden_variables, mechanisms, predictions,
                     theories, accepted_preds, t_start) -> dict:
    """Assemble the final discovery report with full scientific detail."""
    total_time = time.time() - t_start
    report["duration_seconds"] = round(total_time, 1)
    report["completed_at"] = datetime.now().isoformat()
    phases = report.get("phases", {})

    L = []  # summary lines

    L.append("=" * 70)
    L.append("  DISCOVERY REPORT")
    L.append("=" * 70)
    L.append(f"  Topic: {report.get('topic', 'N/A')}")
    L.append(f"  Domain: {report.get('domain', 'N/A')}")
    L.append(f"  Mode: {report.get('mode', 'N/A')}")
    L.append(f"  Duration: {total_time:.1f}s")
    L.append("")

    # ── Papers ──
    L.append("─" * 70)
    L.append(f"  PAPERS ({len(papers)})")
    L.append("─" * 70)
    for p in papers[:10]:
        if isinstance(p, dict):
            src = p.get("source", "?")
            title = p.get("title", "N/A")[:80]
            authors = p.get("authors", "")
            if isinstance(authors, list):
                authors = ", ".join(authors[:3])
            year = p.get("year", "")
            L.append(f"  [{src}] {title}")
            if authors:
                L.append(f"    Authors: {str(authors)[:80]}")
            if year:
                L.append(f"    Year: {year}")
    if len(papers) > 10:
        L.append(f"  ... and {len(papers) - 10} more papers")
    L.append("")

    # ── Knowledge Graph ──
    L.append("─" * 70)
    L.append("  KNOWLEDGE GRAPH")
    L.append("─" * 70)
    L.append(f"  Entities: {len(graph.entities)}")
    L.append(f"  Relationships: {len(graph.relationships)}")
    # Top entities by degree
    if hasattr(graph, 'entities') and graph.entities:
        try:
            degrees = {}
            for src, tgt, _ in graph.relationships:
                degrees[src] = degrees.get(src, 0) + 1
                degrees[tgt] = degrees.get(tgt, 0) + 1
            top_entities = sorted(degrees.items(), key=lambda x: -x[1])[:8]
            if top_entities:
                L.append(f"  Top entities by connections:")
                for name, deg in top_entities:
                    L.append(f"    {name} (degree: {deg})")
        except Exception:
            pass
    L.append("")

    # ── Knowledge Gaps ──
    if gaps:
        L.append("─" * 70)
        L.append(f"  KNOWLEDGE GAPS ({len(gaps)})")
        L.append("─" * 70)
        # Categorize gaps
        gap_types = {}
        for g in gaps:
            gtype = g.get("type", "unknown")
            gap_types.setdefault(gtype, []).append(g)
        for gtype, glist in gap_types.items():
            L.append(f"  [{gtype}] ({len(glist)} gaps)")
            for g in glist[:3]:
                reason = g.get("reason", g.get("description", ""))
                if reason:
                    L.append(f"    - {str(reason)[:120]}")
        L.append("")

    # ── Anomalies ──
    if anomalies:
        L.append("─" * 70)
        L.append(f"  ANOMALIES ({len(anomalies)})")
        L.append("─" * 70)
        for a in anomalies[:8]:
            atype = a.get("type", "?")
            areason = a.get("reason", a.get("description", ""))
            L.append(f"  [{atype}] {str(areason)[:120]}")
        L.append("")

    # ── Hidden Variables ──
    if hidden_variables:
        L.append("─" * 70)
        L.append(f"  HIDDEN VARIABLES ({len(hidden_variables)})")
        L.append("─" * 70)
        for i, hv in enumerate(hidden_variables):
            name = hv.get("name", "?")
            vtype = hv.get("type", "entity")
            desc = hv.get("description", "")
            evidence = hv.get("evidence", "")
            testability = hv.get("testability", "")
            key_params = hv.get("key_parameters", [])
            L.append(f"  {i+1}. [{vtype}] {name}")
            if desc:
                L.append(f"     Description: {desc[:200]}")
            if evidence:
                L.append(f"     Evidence: {str(evidence)[:150]}")
            if testability:
                L.append(f"     Testability: {str(testability)[:120]}")
            # Show derivation if available
            derivation = hv.get("derivation", "")
            if derivation and derivation.lower() not in ("not derivable", "n/a", ""):
                L.append(f"     Derivation:")
                for dline in str(derivation).split("\n")[:4]:
                    L.append(f"       {dline.strip()[:120]}")
            if key_params:
                for kp in key_params[:3]:
                    if isinstance(kp, dict):
                        pname = kp.get("name", "")
                        pval = kp.get("value", kp.get("expected_value", kp.get("range", "")))
                        punits = kp.get("units", "")
                        psource = kp.get("source", kp.get("origin", ""))
                        pdetail = kp.get("source_detail", "")
                        # Epistemic status label
                        label_map = {"cited": "📚 CITED", "derived": "🔧 DERIVED", "estimated": "⚠ ESTIMATED"}
                        label = label_map.get(psource, f"? {psource}" if psource else "")
                        L.append(f"     Parameter: {pname} = {pval} {punits}  {label}")
                        if pdetail:
                            L.append(f"       Source: {str(pdetail)[:120]}")
            L.append("")
        L.append("")

    # ── Mechanisms ──
    if mechanisms:
        L.append("─" * 70)
        L.append(f"  MECHANISMS ({len(mechanisms)})")
        L.append("─" * 70)
        for i, m in enumerate(mechanisms):
            mtype = m.get("type", "mechanism")
            mname = m.get("name", "?")
            mdesc = m.get("description", "")
            inputs = m.get("inputs", [])
            outputs = m.get("outputs", [])
            observables = m.get("observables", [])
            steps = m.get("steps", [])
            L.append(f"  {i+1}. [{mtype}] {mname}")
            if mdesc:
                L.append(f"     {mdesc[:200]}")
            if inputs:
                L.append(f"     Inputs: {', '.join(str(x) for x in (list(inputs) if isinstance(inputs, dict) else inputs)[:5])}")
            if outputs:
                L.append(f"     Outputs: {', '.join(str(x) for x in (list(outputs) if isinstance(outputs, dict) else outputs)[:5])}")
            if observables:
                L.append(f"     Observables: {', '.join(str(x) for x in (list(observables) if isinstance(observables, dict) else observables)[:5])}")
            for j, step in enumerate(steps):
                s_str = str(step)[:150] if step else ""
                if s_str:
                    L.append(f"     Step {j+1}: {s_str}")
            # Show derivation if available
            derivation = m.get("derivation", "")
            if derivation and derivation.lower() not in ("not derivable", "n/a", ""):
                L.append(f"     Derivation:")
                for dline in str(derivation).split("\n")[:5]:
                    L.append(f"       {dline.strip()[:120]}")
            # Show classification (known synthesis vs new physics)
            classification = m.get("classification", "")
            if classification:
                class_labels = {
                    "new_synthesis": "NEW SYNTHESIS of existing data",
                    "new_physics": "NEW PHYSICS — novel mechanism",
                    "new_context_for_known": "KNOWN mechanism in NEW context",
                    "replication": "REPLICATION of known results",
                }
                L.append(f"     Classification: {class_labels.get(classification, classification)}")
            # Show key parameters with epistemic labels
            mech_params = m.get("key_parameters", [])
            if mech_params:
                L.append(f"     Key parameters:")
                for kp in mech_params[:4]:
                    if isinstance(kp, dict):
                        kpname = kp.get("name", "")
                        kpval = kp.get("expected_value", kp.get("value", kp.get("range", "")))
                        kpunits = kp.get("units", "")
                        kpsource = kp.get("source", "")
                        kpdetail = kp.get("source_detail", "")
                        label_map = {"cited": "CITED", "derived": "DERIVED", "estimated": "ESTIMATED"}
                        label = label_map.get(kpsource, kpsource if kpsource else "")
                        kpline = f"       {kpname} = {kpval} {kpunits}"
                        if label:
                            kpline += f"  [{label}]"
                        L.append(kpline)
                        if kpdetail:
                            L.append(f"         {str(kpdetail)[:100]}")
            L.append("")
        L.append("")

    # ── Predictions ──
    if predictions or accepted_preds:
        preds_to_show = accepted_preds if accepted_preds else predictions
        L.append("─" * 70)
        L.append(f"  PREDICTIONS ({len(predictions)} generated, {len(accepted_preds)} accepted)")
        L.append("─" * 70)
        for i, p in enumerate(preds_to_show):
            if isinstance(p, dict):
                ptype = p.get("type", "prediction")
                stmt = p.get("statement", p.get("description", ""))
                conf = p.get("confidence", "")
                test = p.get("test", p.get("test_method", ""))
                fals = p.get("falsification", p.get("falsification_criterion", ""))
                L.append(f"  {i+1}. [{ptype}] {stmt[:180]}")
                if conf:
                    L.append(f"     Confidence: {conf}")
                if test:
                    L.append(f"     Test: {str(test)[:120]}")
                if fals:
                    L.append(f"     Falsification: {str(fals)[:120]}")
                L.append("")
        L.append("")

    # ── Theory Competition ──
    if theories:
        L.append("─" * 70)
        tc = phases.get("theory_competition", {})
        total_compared = tc.get("theories_compared", len(theories))
        eliminated_count = len(tc.get("eliminated", []))
        L.append(f"  THEORY TOURNAMENT ({total_compared} entered, {len(theories)} survived, {eliminated_count} eliminated)")
        L.append("─" * 70)

        # Tournament log
        tournament_log = tc.get("tournament_log", [])
        if tournament_log:
            L.append(f"  Tournament rounds:")
            for entry in tournament_log:
                round_num = entry.get("round", "?")
                action = entry.get("action", "?")
                survived = entry.get("survived", entry.get("candidates", "?"))
                elim = entry.get("eliminated", "")
                if elim:
                    L.append(f"    Round {round_num}: {action} — {survived} survived, {elim} eliminated")
                else:
                    L.append(f"    Round {round_num}: {action} — {survived} candidates")

        # Competition analysis
        analysis = tc.get("competition_analysis", "")
        if analysis:
            L.append(f"\n  Analysis: {str(analysis)[:300]}")

        # Sort by overall score
        scored = []
        for t in theories:
            if isinstance(t, dict):
                score = t.get("scores", {}).get("overall", 0)
                scored.append((score, t))
        scored.sort(key=lambda x: -x[0])

        L.append(f"\n  Survivors (ranked):")
        for rank, (score, t) in enumerate(scored[:7], 1):
            tname = t.get("name", "?")
            tdesc = t.get("description", "")
            ttype = t.get("type", "hypothesis")
            scores = t.get("scores", {})
            mechanism = t.get("mechanism", "")
            explains = t.get("explains", [])
            fails = t.get("fails_to_explain", [])
            assumptions = t.get("key_assumptions", [])
            causal = t.get("causal_status", "")

            L.append(f"  {rank}. {tname} (score: {score:.2f}) [{ttype}]")
            if tdesc:
                L.append(f"     {tdesc[:200]}")
            if mechanism:
                L.append(f"     Mechanism: {str(mechanism)[:150]}")
            if explains:
                L.append(f"     Explains: {'; '.join(str(e)[:60] for e in explains[:3])}")
            if fails:
                L.append(f"     Fails to explain: {'; '.join(str(f)[:60] for f in fails[:3])}")
            if assumptions:
                L.append(f"     Key assumptions: {'; '.join(str(a)[:60] for a in assumptions[:3])}")
            if causal:
                L.append(f"     Causal status: {causal}")
            # Per-dimension scores
            if scores and isinstance(scores, dict):
                dims = []
                for dim in ["explanatory_power", "predictive_power", "falsifiability",
                            "evidence_strength", "novelty", "simplicity", "mathematical_rigor"]:
                    if dim in scores:
                        dims.append(f"{dim}={scores[dim]:.0%}" if isinstance(scores[dim], float) and scores[dim] <= 1 else f"{dim}={scores[dim]}")
                if dims:
                    L.append(f"     Scores: {', '.join(dims)}")
            L.append("")

        # Discriminating experiments
        experiments = tc.get("discriminating_experiments", [])
        if experiments:
            L.append(f"  Discriminating experiments (would separate top 2):")
            for i, exp in enumerate(experiments[:3], 1):
                L.append(f"    {i}. {str(exp)[:150]}")
            L.append("")

        # Show eliminated theories briefly
        eliminated = tc.get("eliminated", [])
        if eliminated:
            L.append(f"  Eliminated ({len(eliminated)}):")
            for t in eliminated[:5]:
                tname = t.get("name", "?")
                score = t.get("scores", {}).get("overall", 0)
                reason = t.get("elimination_reason", "")
                L.append(f"    ✗ {tname} (score: {score:.2f}) — {str(reason)[:80]}")
            if len(eliminated) > 5:
                L.append(f"    ... and {len(eliminated) - 5} more")
        L.append("")

    # ── Adversarial Test Results ──
    adv_test = phases.get("adversarial_test", {})
    if adv_test:
        L.append("─" * 70)
        L.append("  ADVERSARIAL TEST — What Survived the Attack")
        L.append("─" * 70)
        summary = adv_test.get("survival_summary", {})
        for category in ["hidden_variables", "mechanisms", "theories"]:
            s = summary.get(category, {})
            started = s.get("started", 0)
            survived = s.get("survived", 0)
            weakened = s.get("weakened", 0)
            killed = s.get("killed", 0)
            if started > 0:
                L.append(f"  {category}: {started} tested → {survived} survived, {weakened} weakened, {killed} killed")

        # Show individual test results
        for test_list_name, label in [
            ("hidden_variable_tests", "Hidden Variables"),
            ("mechanism_tests", "Mechanisms"),
            ("theory_tests", "Theories"),
        ]:
            test_list = adv_test.get(test_list_name, [])
            if test_list:
                L.append(f"  {label}:")
                for t in test_list:
                    name = t.get("name", "?")
                    verdict = t.get("verdict", "?")
                    score = t.get("attack_score", 0)
                    existing = t.get("existing_theory", "")
                    can_remove = t.get("can_remove_variables", False)
                    falsification = t.get("falsification", "")
                    reasoning = t.get("reasoning", "")
                    icon = {"survived": "✓", "weakened": "⚠", "killed": "✗"}.get(verdict, "?")

                    L.append(f"    {icon} {name[:50]} [{verdict}, attack score: {score:.2f}]")
                    if existing and existing.lower() not in ("none known", "no known theory explains this", ""):
                        L.append(f"      Existing theory: {str(existing)[:100]}")
                    if can_remove:
                        removable = t.get("removable_variables", [])
                        L.append(f"      Can remove: {', '.join(str(v) for v in removable[:3])}")
                    if falsification and falsification.lower() not in ("none", "test failed", ""):
                        L.append(f"      Falsification: {str(falsification)[:100]}")
                    if reasoning:
                        L.append(f"      Verdict: {str(reasoning)[:120]}")
        L.append("")

    # ── Critical Evaluation ──
    pr = phases.get("peer_review", {})
    if pr:
        L.append("─" * 70)
        L.append("  CRITICAL EVALUATION — 6-Dimension Assessment")
        L.append("─" * 70)
        L.append(f"  Overall Score: {pr.get('overall_score', 0)}/10")
        L.append(f"  Recommendation: {pr.get('recommendation', 'unknown')}")
        summary = pr.get("summary", "")
        if summary:
            L.append(f"  Summary: {str(summary)[:300]}")
        # Dimension scores
        for dim in ["novelty", "methodology", "significance", "clarity", "limitations", "reproducibility"]:
            dim_data = pr.get(dim, {})
            if isinstance(dim_data, dict):
                score = dim_data.get("score", "?")
                comment = dim_data.get("comment", "")
                L.append(f"  {dim.capitalize()}: {score}/10 — {str(comment)[:80]}")
        # Issues
        major = pr.get("major_issues", [])
        minor = pr.get("minor_issues", [])
        questions = pr.get("questions_for_authors", [])
        if major:
            L.append(f"  Major issues ({len(major)}):")
            for issue in major:
                L.append(f"    ! {str(issue)[:120]}")
        if minor:
            L.append(f"  Minor issues ({len(minor)}):")
            for issue in minor:
                L.append(f"    - {str(issue)[:120]}")
        if questions:
            L.append(f"  Questions for authors:")
            for q in questions:
                L.append(f"    ? {str(q)[:120]}")
        L.append("")

    # ── Contradictions ──
    contradictions = phases.get("contradictions", {})
    contradiction_list = contradictions.get("contradictions", [])
    if contradiction_list:
        L.append("─" * 70)
        L.append(f"  CONTRADICTIONS ({len(contradiction_list)})")
        L.append("─" * 70)
        for i, c in enumerate(contradiction_list[:5]):
            if isinstance(c, dict):
                cdesc = c.get("description", c.get("detail", ""))
                csev = c.get("severity", "")
                L.append(f"  {i+1}. {str(cdesc)[:150]}")
                if csev:
                    L.append(f"     Severity: {csev}")
            elif isinstance(c, str):
                L.append(f"  {i+1}. {c[:150]}")
        L.append("")

    # ── Computational Verification ──
    comp = phases.get("computational_verification", {})
    if comp:
        L.append("─" * 70)
        L.append("  COMPUTATIONAL VERIFICATION")
        L.append("─" * 70)
        L.append(f"  Computations run: {comp.get('total_computations', 0)}")
        L.append(f"  Support level: {comp.get('support_level', 'unknown')}")
        for finding in comp.get("key_findings", [])[:5]:
            L.append(f"    → {str(finding)[:120]}")
        L.append("")

    # ── Skeptic Review ──
    sr = phases.get("skeptic_review", {})
    if sr:
        L.append("─" * 70)
        L.append("  SKEPTIC REVIEW")
        L.append("─" * 70)
        L.append(f"  Recommendation: {sr.get('recommendation', 'N/A')}")
        L.append(f"  Revised confidence: {sr.get('revised_confidence', 0):.0%}")
        strengths = sr.get("strengths", [])
        weaknesses = sr.get("weaknesses", [])
        if strengths:
            L.append(f"  Strengths ({len(strengths)}):")
            for s in strengths:
                L.append(f"    + {str(s)[:150]}")
        if weaknesses:
            L.append(f"  Weaknesses ({len(weaknesses)}):")
            for w in weaknesses:
                L.append(f"    - {str(w)[:150]}")
        failure_conditions = sr.get("failure_conditions", [])
        if failure_conditions:
            L.append(f"  Failure conditions:")
            for fc in failure_conditions:
                L.append(f"    ! {str(fc)[:120]}")
        L.append("")

    # ── Discovery Scoring ──
    ds = phases.get("discovery_scoring", {})
    if ds:
        L.append("─" * 70)
        L.append("  DISCOVERY SCORING")
        L.append("─" * 70)
        L.append(f"  Overall Score: {ds.get('discovery_score', 0):.0f}/100")
        L.append(f"  Grade: {ds.get('grade', 'F')}")
        scores = ds.get("scores", {})
        if scores:
            L.append(f"  Dimension scores:")
            for dim, val in scores.items():
                L.append(f"    {dim}: {val}")
        strengths = ds.get("strengths", [])
        weaknesses = ds.get("weaknesses", [])
        if strengths:
            L.append(f"  Strengths: {'; '.join(str(s)[:80] for s in strengths[:3])}")
        if weaknesses:
            L.append(f"  Weaknesses: {'; '.join(str(w)[:80] for w in weaknesses[:3])}")
        summary = ds.get("summary", "")
        if summary:
            L.append(f"  Summary: {str(summary)[:200]}")
        L.append("")

    # ── Discovery Classification — What's Known vs What's New ──
    dc = phases.get("discovery_classification", {})
    if dc:
        L.append("─" * 70)
        L.append("  DISCOVERY CLASSIFICATION — What's Known vs What's New")
        L.append("─" * 70)
        classification = dc.get('classification', 'N/A')
        class_desc = {
            "replication": "Confirms existing knowledge — validates what was already known",
            "synthesis": "Combines existing ideas in a new way — new connections between known concepts",
            "extension": "New application of known mechanisms — applies existing physics to new domain",
            "novel_theory": "Requires new mechanism, new prediction, new mathematics — genuinely new",
        }
        L.append(f"  Classification: {classification}")
        L.append(f"  Meaning: {class_desc.get(classification, 'Unknown classification')}")
        L.append(f"  Details:")
        L.append(f"    New mechanism proposed: {dc.get('has_new_mechanism', False)}")
        L.append(f"    New prediction generated: {dc.get('has_new_prediction', False)}")
        L.append(f"    New mathematics introduced: {dc.get('has_new_math', False)}")
        L.append(f"    Not found in literature: {dc.get('not_in_literature', False)}")

        # Summarize what's known vs what's new from mechanisms
        mech_classifications = {}
        for m in mechanisms:
            mc = m.get("classification", "")
            if mc:
                mech_classifications[mc] = mech_classifications.get(mc, 0) + 1
        if mech_classifications:
            L.append(f"  Mechanism breakdown:")
            for mc, count in mech_classifications.items():
                mc_label = {
                    "new_synthesis": "New synthesis of existing data",
                    "new_physics": "New physics — novel mechanism",
                    "new_context_for_known": "Known mechanism in new context",
                    "replication": "Replication of known results",
                }.get(mc, mc)
                L.append(f"    {mc_label}: {count}")
        L.append("")

    # ── Errors ──
    if report.get("errors"):
        L.append("─" * 70)
        L.append(f"  ERRORS ({len(report['errors'])})")
        L.append("─" * 70)
        for e in report["errors"]:
            L.append(f"  ! {e}")
        L.append("")

    L.append("=" * 70)

    report_text = "\n".join(L)
    report["summary_text"] = report_text

    try:
        print(report_text)
    except UnicodeEncodeError:
        import sys
        safe = report_text.encode("ascii", errors="replace").decode("ascii")
        sys.stdout.buffer.write(safe.encode("ascii", errors="replace") + b"\n")

    # Save report + text + update dashboard index
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    report_path = data_dir / f"discovery_v2_{int(time.time())}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\n  Report saved: {report_path}")

    # Save human-readable text report alongside JSON
    txt_path = report_path.with_suffix(".txt")
    txt_path.write_text(report_text, encoding="utf-8")
    print(f"  Text report: {txt_path}")

    # Update reports index for dashboard
    idx_path = data_dir / "reports_index.json"
    try:
        idx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else []
    except Exception:
        idx = []
    rel = str(report_path.relative_to(Path(__file__).parent.parent))
    if rel not in idx:
        idx.append(rel)
        idx_path.write_text(json.dumps(idx[-50:], indent=2), encoding="utf-8")

    return report


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "KRAS G12C inhibitor resistance"
    run_discovery_pipeline(topic, mode="full")