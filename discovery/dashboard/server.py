#!/usr/bin/env python3
"""
Luka Dashboard Server — serves the discovery dashboard with live API endpoints.

Usage:
    python discovery/dashboard/server.py              # default port 8899
    python discovery/dashboard/server.py --port 9000  # custom port

Endpoints:
    GET /                    → Dashboard HTML
    GET /api/reports         → List all discovery reports
    GET /api/latest          → Latest report (full JSON)
    GET /api/report/<id>     → Specific report by timestamp
    GET /api/graph           → Knowledge graph (entities + relationships)
    GET /api/health          → System health
    GET /api/status          → Pipeline status
    GET /api/memory          → Hypothesis memory
    GET /api/phases          → Phase breakdown of latest report
    GET /api/scoring         → Scoring details of latest report
    GET /api/adversarial     → Adversarial test results
    GET /api/derivations     → Derivation engine output
    GET /api/peer-review     → Peer review results
    GET /api/refinement      → Refinement pipeline results
"""

import json
import os
import sys
import time
import glob
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # luka root
DATA_DIR = BASE_DIR / "data"
DASHBOARD_DIR = Path(__file__).resolve().parent


def load_reports():
    """Load all discovery reports from data/ directory."""
    reports = []
    # Try index first
    idx_path = DATA_DIR / "reports_index.json"
    if idx_path.exists():
        try:
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
            for rel_path in idx[-50:]:
                full = BASE_DIR / rel_path
                if full.exists():
                    try:
                        reports.append(json.loads(full.read_text(encoding="utf-8")))
                    except Exception:
                        pass
        except Exception:
            pass

    # Fallback: scan for discovery_*.json files
    if not reports:
        for f in sorted(DATA_DIR.glob("discovery_*.json")):
            if "archive" in f.name:
                continue
            try:
                reports.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass

    return reports


def load_latest():
    """Load the most recent report."""
    reports = load_reports()
    return reports[-1] if reports else None


def load_graph():
    """Load knowledge graph."""
    for path in [
        DATA_DIR / "knowledge_graph.json",
        BASE_DIR / "discovery" / "graph" / "knowledge_graph.json",
    ]:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {"entities": {}, "relationships": [], "papers": {}}


def load_memory():
    """Load hypothesis memory."""
    for path in [
        BASE_DIR / "discovery" / "hypotheses" / "active.json",
        DATA_DIR / "hypothesis_memory.json",
    ]:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return []


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves dashboard + API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]

        # API endpoints
        if path.startswith("/api/"):
            self._handle_api(path)
            return

        # Serve dashboard HTML
        if path == "/" or path == "/index.html":
            self.path = "/index.html"

        super().do_GET()

    def _handle_api(self, path):
        """Handle API requests."""
        data = None

        if path == "/api/reports":
            reports = load_reports()
            # Return summary of each report (not full JSON)
            data = []
            for r in reports:
                phases = r.get("phases", {})
                sc = phases.get("discovery_scoring", {})
                data.append({
                    "topic": r.get("topic", "?"),
                    "domain": r.get("domain", "?"),
                    "score": sc.get("discovery_score", 0),
                    "grade": sc.get("grade", "F"),
                    "unified_score": sc.get("unified_score"),
                    "papers": phases.get("literature", {}).get("papers_found", 0),
                    "entities": phases.get("knowledge_graph", {}).get("entities", 0),
                    "duration": r.get("duration_seconds", 0),
                    "timestamp": r.get("completed_at", r.get("started_at", "")),
                })

        elif path == "/api/latest":
            data = load_latest()

        elif path.startswith("/api/report/"):
            ts = path.split("/")[-1]
            reports = load_reports()
            for r in reports:
                if ts in str(r.get("completed_at", "")) or ts in str(r.get("started_at", "")):
                    data = r
                    break

        elif path == "/api/graph":
            data = load_graph()

        elif path == "/api/health":
            data = {
                "status": "ok",
                "reports": len(load_reports()),
                "graph_entities": len(load_graph().get("entities", {})),
                "data_dir": str(DATA_DIR),
                "uptime": time.time(),
            }

        elif path == "/api/status":
            latest = load_latest()
            if latest:
                phases = latest.get("phases", {})
                data = {
                    "topic": latest.get("topic", "?"),
                    "domain": latest.get("domain", "?"),
                    "phases_completed": len(phases),
                    "score": phases.get("discovery_scoring", {}).get("discovery_score", 0),
                    "grade": phases.get("discovery_scoring", {}).get("grade", "F"),
                    "duration": latest.get("duration_seconds", 0),
                }
            else:
                data = {"status": "no reports"}

        elif path == "/api/memory":
            data = load_memory()

        elif path == "/api/phases":
            latest = load_latest()
            if latest:
                phases = latest.get("phases", {})
                data = {}
                for k, v in phases.items():
                    if isinstance(v, dict):
                        # Summarize each phase
                        summary = {}
                        for sk, sv in v.items():
                            if isinstance(sv, (int, float, str, bool)):
                                summary[sk] = sv
                            elif isinstance(sv, list):
                                summary[sk + "_count"] = len(sv)
                        data[k] = summary
                    else:
                        data[k] = v

        elif path == "/api/scoring":
            latest = load_latest()
            if latest:
                data = latest.get("phases", {}).get("discovery_scoring", {})

        elif path == "/api/adversarial":
            latest = load_latest()
            if latest:
                data = latest.get("phases", {}).get("adversarial_test", {})

        elif path == "/api/derivations":
            latest = load_latest()
            if latest:
                data = latest.get("phases", {}).get("derivation_engine", {})

        elif path == "/api/peer-review":
            latest = load_latest()
            if latest:
                data = latest.get("phases", {}).get("peer_review", {})

        elif path == "/api/refinement":
            latest = load_latest()
            if latest:
                data = latest.get("refinement", {})

        else:
            self.send_error(404, f"Unknown API endpoint: {path}")
            return

        self._send_json(data)

    def _send_json(self, data):
        """Send JSON response with CORS headers."""
        body = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress request logging unless verbose."""
        if "--verbose" in sys.argv or "-v" in sys.argv:
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Luka Discovery Dashboard Server")
    parser.add_argument("--port", type=int, default=8899, help="Port (default: 8899)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), DashboardHandler)
    print(f"\n  Luka Dashboard Server")
    print(f"  =====================")
    print(f"  URL:  http://{args.host}:{args.port}")
    print(f"  Data: {DATA_DIR}")
    print(f"  Reports: {len(load_reports())}")
    print(f"\n  API Endpoints:")
    print(f"    GET /api/reports      — All reports (summary)")
    print(f"    GET /api/latest       — Latest report (full)")
    print(f"    GET /api/graph        — Knowledge graph")
    print(f"    GET /api/phases       — Phase breakdown")
    print(f"    GET /api/scoring      — Scoring details")
    print(f"    GET /api/adversarial  — Adversarial results")
    print(f"    GET /api/derivations  — Derivation engine")
    print(f"    GET /api/peer-review  — Peer review")
    print(f"    GET /api/refinement   — Refinement results")
    print(f"    GET /api/memory       — Hypothesis memory")
    print(f"\n  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()