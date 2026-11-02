"""Run simulator + create readable report."""
import json, sys
sys.path.insert(0, r"C:\Users\Admin\Desktop\luka")
from discovery.scientific_simulator import ScientificSimulator

with open(r"C:\Users\Admin\Desktop\luka\data\discovery_What_microphysical_process_absent_from_current_MAT_1783109470.json", "r", encoding="utf-8") as f:
    data = json.load(f)

phases = data.get("phases", {})
mechanisms = phases.get("mechanism_generation", {}).get("mechanism_details", [])
hidden_vars = phases.get("missing_variables", {}).get("variable_details", [])
predictions = phases.get("prediction_engine", {}).get("accepted_details", [])
topic = data.get("topic", "")
domain = data.get("domain", "")

sim = ScientificSimulator()
results = sim.simulate(
    mechanisms=mechanisms[:5],
    hidden_variables=hidden_vars[:5],
    predictions=predictions[:5],
    topic=topic,
    domain=domain,
)

# Save sim results
with open(r"C:\Users\Admin\Desktop\luka\sim_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Build readable report
lines = []
lines.append("=" * 80)
lines.append("LUKA DISCOVERY REPORT")
lines.append("=" * 80)
lines.append("")
lines.append(f"Topic: {topic}")
lines.append(f"Domain: {domain}")
lines.append(f"Mode: {data.get('mode','')}")
lines.append(f"Started: {data.get('started_at','')}")
lines.append("")

# Literature
lit = phases.get("literature", {})
lines.append("-" * 80)
lines.append("SECTION 1: LITERATURE REVIEW")
lines.append("-" * 80)
lines.append(f"Papers found: {lit.get('papers_found', 0)}")
lines.append(f"Sources: {', '.join(lit.get('sources', []))}")
lines.append("")
for i, p in enumerate(lit.get("papers", [])[:20], 1):
    title = p.get("title", "N/A")
    authors = ", ".join(p.get("authors", [])[:3])
    year = p.get("year", "?")
    cites = p.get("citation_count", 0)
    lines.append(f"  [{i}] {title}")
    lines.append(f"      Authors: {authors}")
    lines.append(f"      Year: {year} | Citations: {cites}")
    lines.append("")

# Knowledge Graph
kg = phases.get("knowledge_graph", {})
lines.append("-" * 80)
lines.append("SECTION 2: KNOWLEDGE GRAPH")
lines.append("-" * 80)
lines.append(f"Entities: {kg.get('entities', 0)}")
lines.append(f"Relationships: {kg.get('relationships', 0)}")
lines.append("")

# Gaps
gaps = phases.get("gap_detection", {})
lines.append("-" * 80)
lines.append("SECTION 3: KNOWLEDGE GAPS")
lines.append("-" * 80)
lines.append(f"Total gaps: {gaps.get('total_gaps', 0)}")
for g in gaps.get("top_gaps", [])[:5]:
    lines.append(f"  - [{g.get('type','')}] {g.get('description','')[:120]}")
lines.append("")

# Anomalies
anom = phases.get("anomaly_detection", {})
lines.append("-" * 80)
lines.append("SECTION 4: ANOMALIES")
lines.append("-" * 80)
lines.append(f"Total anomalies: {anom.get('total_anomalies', 0)}")
for a in anom.get("top_anomalies", [])[:5]:
    lines.append(f"  - [{a.get('type','')}] {a.get('description','')[:120]}")
lines.append("")

# Curious Questioning
cq = phases.get("curious_questioning", {})
lines.append("-" * 80)
lines.append("SECTION 5: CURIOUS QUESTIONING (THE NEWTON STEP)")
lines.append("-" * 80)
lines.append(f"Core question: {cq.get('core_question','')}")
obs_count = cq.get('observations', 0)
q_count = cq.get('questions', 0)
lines.append(f"Observations: {obs_count if isinstance(obs_count, int) else len(obs_count)}")
lines.append(f"Questions: {q_count if isinstance(q_count, int) else len(q_count)}")
lines.append("")

# Hidden Variables
hv = phases.get("missing_variables", {})
lines.append("-" * 80)
lines.append("SECTION 6: HIDDEN VARIABLES")
lines.append("-" * 80)
for v in hv.get("variable_details", []):
    lines.append(f"  Name: {v.get('name','')}")
    lines.append(f"  Type: {v.get('type','')}")
    lines.append(f"  Description: {v.get('description','')[:200]}")
    mf = v.get("mathematical_formalism", "")
    if mf:
        lines.append(f"  Math: {mf[:200]}")
    lines.append("")

# Mechanisms
mg = phases.get("mechanism_generation", {})
lines.append("-" * 80)
lines.append("SECTION 7: MECHANISMS")
lines.append("-" * 80)
lines.append(f"Generated: {mg.get('mechanisms_generated', 0)}")
for m in mg.get("mechanism_details", []):
    lines.append(f"  Name: {m.get('name','')}")
    lines.append(f"  Type: {m.get('type','')}")
    lines.append(f"  Description: {m.get('description','')[:300]}")
    steps = m.get("steps", [])
    if steps:
        lines.append(f"  Steps:")
        for i, s in enumerate(steps[:5], 1):
            if isinstance(s, dict):
                lines.append(f"    Step {i}: {s.get('text', str(s))[:150]}")
            else:
                lines.append(f"    Step {i}: {str(s)[:150]}")
    params = m.get("key_parameters", [])
    if params:
        lines.append(f"  Key parameters:")
        for p in params[:5]:
            if isinstance(p, dict):
                lines.append(f"    - {p.get('name','')}: {p.get('expected_value','')} [{p.get('source','')}]")
    lines.append("")

# Predictions
pe = phases.get("prediction_engine", {})
lines.append("-" * 80)
lines.append("SECTION 8: PREDICTIONS")
lines.append("-" * 80)
lines.append(f"Generated: {pe.get('total_predictions', 0)} | Accepted: {pe.get('accepted', 0)}")
for p in pe.get("accepted_details", []):
    lines.append(f"  Type: {p.get('type','')}")
    lines.append(f"  Statement: {p.get('statement','')[:250]}")
    lines.append(f"  Confidence: {p.get('confidence','')}")
    lines.append(f"  Test: {p.get('test_method','')[:200]}")
    lines.append(f"  Falsification: {p.get('falsification','')[:200]}")
    lines.append("")

# Theory Competition
tc = phases.get("theory_competition", {})
lines.append("-" * 80)
lines.append("SECTION 9: THEORY TOURNAMENT")
lines.append("-" * 80)
lines.append(f"Theories compared: {tc.get('theories_compared', 0)}")
lines.append(f"Winner: {tc.get('winner_name','')} (score: {tc.get('winner_score','')})")
for t in tc.get("theories", []):
    lines.append(f"  - {t.get('name','')} (score: {t.get('score','')})")
lines.append("")

# Adversarial Test
at = phases.get("adversarial_test", {})
lines.append("-" * 80)
lines.append("SECTION 10: ADVERSARIAL TEST")
lines.append("-" * 80)
summary = at.get("survival_summary", {})
lines.append(f"Hidden variables: {summary.get('hidden_variables','')}")
lines.append(f"Mechanisms: {summary.get('mechanisms','')}")
lines.append(f"Theories: {summary.get('theories','')}")
lines.append("")

# Peer Review
pr = phases.get("peer_review", {})
lines.append("-" * 80)
lines.append("SECTION 11: PEER REVIEW (CRITICAL EVALUATION)")
lines.append("-" * 80)
lines.append(f"Overall score: {pr.get('overall_score','')}/10")
lines.append(f"Recommendation: {pr.get('recommendation','')}")
lines.append(f"Summary: {pr.get('summary','')[:500]}")
lines.append("")
lines.append("Major issues:")
for issue in pr.get("major_issues", []):
    lines.append(f"  ! {issue}")
lines.append("")
lines.append("Minor issues:")
for issue in pr.get("minor_issues", []):
    lines.append(f"  - {issue}")
lines.append("")

# Skeptic Review
sr = phases.get("skeptic_review", {})
lines.append("-" * 80)
lines.append("SECTION 12: SKEPTIC REVIEW")
lines.append("-" * 80)
lines.append(f"Recommendation: {sr.get('recommendation','')}")
lines.append(f"Confidence: {sr.get('revised_confidence','')}")
lines.append("Strengths:")
for s in sr.get("strengths", []):
    lines.append(f"  + {s}")
lines.append("Weaknesses:")
for w in sr.get("weaknesses", []):
    lines.append(f"  - {w}")
lines.append("")

# Discovery Scoring
ds = phases.get("discovery_scoring", {})
lines.append("-" * 80)
lines.append("SECTION 13: DISCOVERY SCORING")
lines.append("-" * 80)
lines.append(f"Score: {ds.get('discovery_score','')}/100")
lines.append(f"Grade: {ds.get('grade','')}")
lines.append(f"Classification: {phases.get('discovery_classification',{}).get('classification','')}")
lines.append(f"Summary: {ds.get('summary','')}")
lines.append("")

# Scientific Simulator
lines.append("-" * 80)
lines.append("SECTION 14: SCIENTIFIC SIMULATOR (RE-RUN)")
lines.append("-" * 80)
lines.append(f"Equations solved: {len(results.get('equation_results', []))}")
lines.append(f"Parameter sweeps: {len(results.get('parameter_sweeps', []))}")
lines.append(f"Observables: {len(results.get('observables', []))}")
lines.append(f"Consistency checks: {len(results.get('consistency_checks', []))}")
lines.append(f"Tools used: {', '.join(results.get('tools_used', []))}")
lines.append("")
lines.append("Equation results:")
for e in results.get("equation_results", []):
    expr = e.get("expression", e.get("value", "?"))
    lines.append(f"  {e.get('variable','?')} = {expr} [{e.get('status','')}]")
lines.append("")
lines.append("Parameter sweeps:")
for s in results.get("parameter_sweeps", []):
    val = s.get("value", s.get("range", "?"))
    lines.append(f"  {s.get('parameter','?')} = {val} [{s.get('status','')}]")
lines.append("")
lines.append("Consistency checks:")
for c in results.get("consistency_checks", []):
    lines.append(f"  {c.get('check','?')}: {c.get('finding','')}")
lines.append("")

# Experimental Validation
ev = phases.get("experimental_validation", {})
lines.append("-" * 80)
lines.append("SECTION 15: EXPERIMENTAL VALIDATION")
lines.append("-" * 80)
for plan in ev.get("plans", []):
    lines.append(f"  Type: {plan.get('type','')}")
    lines.append(f"  Timeline: {plan.get('timeline','')[:200]}")
    lines.append(f"  Cost: {plan.get('cost','')}")
    lines.append("")

# Contradictions
con = phases.get("contradictions", {})
lines.append("-" * 80)
lines.append("SECTION 16: CONTRADICTIONS")
lines.append("-" * 80)
lines.append(f"Total: {con.get('total', 0)}")
for c in con.get("contradictions", []):
    lines.append(f"  - {c.get('description','')[:200]}")
lines.append("")

# Refinement
ref = phases.get("refinement", {})
lines.append("-" * 80)
lines.append("SECTION 17: REFINEMENT")
lines.append("-" * 80)
lines.append(f"Verdict: {ref.get('verdict','')}")
lines.append(f"Grade: {ref.get('grade','')}")
lines.append(f"Score: {ref.get('score','')}")
lines.append("")

# Data Analysis
da = phases.get("data_analysis", {})
lines.append("-" * 80)
lines.append("SECTION 18: DATA ANALYSIS")
lines.append("-" * 80)
lines.append(f"Datasets fetched: {da.get('datasets_fetched', 0)}")
for r in da.get("results", []):
    if isinstance(r, dict):
        lines.append(f"  - {r.get('dataset','')}: {r.get('records',0)} records")
    else:
        lines.append(f"  - {str(r)[:200]}")
lines.append("")

# Open Questions
lines.append("-" * 80)
lines.append("SECTION 19: OPEN QUESTIONS & NEXT STEPS")
lines.append("-" * 80)
gd = phases.get("gap_discovery", {})
for q in gd.get("next_questions", []):
    lines.append(f"  ? {q}")
lines.append("")

# Cross-domain analogies
cross = phases.get("cross_domain_analogies", [])
if cross:
    lines.append("-" * 80)
    lines.append("SECTION 20: CROSS-DOMAIN ANALOGIES")
    lines.append("-" * 80)
    for ca in cross:
        lines.append(f"  Domain A: {ca.get('domain_a','')}")
        lines.append(f"  Domain B: {ca.get('domain_b','')}")
        lines.append(f"  Connection: {ca.get('connection','')[:200]}")
        lines.append(f"  Hypothesis: {ca.get('hypothesis','')[:200]}")
        lines.append("")

lines.append("=" * 80)
lines.append("END OF REPORT")
lines.append("=" * 80)

report = "\n".join(lines)
with open(r"C:\Users\Admin\Desktop\luka_report_readable.txt", "w", encoding="utf-8") as f:
    f.write(report)
print(f"Readable report: {len(lines)} lines written to luka_report_readable.txt")
