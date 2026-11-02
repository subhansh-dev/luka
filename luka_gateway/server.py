"""WebSocket JSON-RPC server dispatching to RumiLive."""
from __future__ import annotations
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import websockets
    from websockets.server import serve as ws_serve
except ImportError:
    print("[gateway] websockets not installed. Run: pip install websockets", file=sys.stderr)
    sys.exit(1)

from luka_gateway.protocol import (
    JsonRpcRequest, JsonRpcResponse, JsonRpcEvent,
    send_message, log_error, set_websocket,
)


class GatewayServer:
    """WebSocket server reading JSON-RPC from clients, dispatching to RumiLive."""

    def __init__(self, host: str = "127.0.0.1", port: int = 18789):
        self._host = host
        self._port = port
        self._running = True
        self._rumi = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session_ready = False
        self._event_queue: asyncio.Queue = asyncio.Queue()

    def start(self):
        """Main entry - run the event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run())

    async def _run(self):
        """Core loop: start WebSocket server."""
        log_error(f"Gateway server starting on ws://{self._host}:{self._port}")

        async with ws_serve(self.handler, self._host, self._port):
            log_error("Gateway ready - waiting for Ink TUI connection")
            await asyncio.Future()  # run forever

    async def handler(self, websocket):
        """Handle a single WebSocket client connection."""
        set_websocket(websocket)
        log_error(f"Client connected: {websocket.remote_address}")

        send_message(JsonRpcEvent(
            method="gateway.ready",
            params={"version": "3.1"},
        ))

        try:
            async for message in websocket:
                line = message if isinstance(message, str) else message.decode("utf-8")
                line = line.strip()
                if not line:
                    continue

                try:
                    request = JsonRpcRequest.from_json(line)
                    await self._handle_request(request)
                except json.JSONDecodeError as e:
                    log_error(f"Bad JSON: {e}")
                except Exception as e:
                    log_error(f"Error: {e}")
        except websockets.ConnectionClosed:
            log_error("Client disconnected")
        finally:
            set_websocket(None)

    async def _handle_request(self, req: JsonRpcRequest):
        """Dispatch a JSON-RPC request."""
        method = req.method
        params = req.params

        try:
            if method == "session.create":
                await self._handle_session_create(req, params)
            elif method == "session.info":
                await self._handle_session_info(req)
            elif method == "session.list":
                await self._handle_session_list(req)
            elif method == "chat.send":
                await self._handle_chat_send(req, params)
            elif method == "chat.interrupt":
                await self._handle_chat_interrupt(req)
            elif method == "slash.execute":
                await self._handle_slash(req, params)
            else:
                send_message(JsonRpcResponse(
                    id=req.id,
                    error={"code": -32601, "message": f"Unknown method: {method}"},
                ))
        except Exception as e:
            log_error(f"Handler error for {method}: {e}")
            send_message(JsonRpcResponse(
                id=req.id,
                error={"code": -32603, "message": str(e)},
            ))

    async def _handle_session_create(self, req: JsonRpcRequest, params: dict):
        """Initialize RumiLive and start the session."""
        sys.path.insert(0, str(_project_root))
        from main import RumiLive

        adapter = GatewayAdapter(self)
        self._rumi = RumiLive(adapter)

        send_message(JsonRpcResponse(
            id=req.id,
            result={
                "session_id": f"luka-{int(time.time())}",
                "model": "gemini-2.5-flash",
                "personality": "professional",
            },
        ))

        asyncio.ensure_future(self._run_rumi())

    async def _run_rumi(self):
        """Run RumiLive's async loop in background."""
        try:
            self._session_ready = True
            send_message(JsonRpcEvent(
                method="system.message",
                params={"content": "Connecting to Gemini..."},
            ))
            await self._rumi.run()
        except Exception as e:
            err_msg = str(e)
            log_error(f"RumiLive error: {err_msg}")
            send_message(JsonRpcEvent(
                method="error",
                params={"code": -1, "message": f"Session error: {err_msg[:200]}"},
            ))

    async def _handle_session_info(self, req: JsonRpcRequest):
        """Return current session metadata."""
        send_message(JsonRpcResponse(
            id=req.id,
            result={
                "model": "gemini-2.5-flash",
                "personality": "professional",
                "uptime": int(time.time() - self._rumi._start_time) if self._rumi else 0,
                "tokens": self._rumi.ui._total_tokens if self._rumi else 0,
                "cost": self._rumi.ui._total_cost if self._rumi else 0,
            },
        ))

    async def _handle_session_list(self, req: JsonRpcRequest):
        """Return available sessions."""
        send_message(JsonRpcResponse(
            id=req.id,
            result={
                "sessions": [
                    {
                        "id": f"luka-{int(time.time())}",
                        "title": "Current Session",
                        "model": "gemini-2.5-flash",
                        "messages": 0,
                    }
                ],
            },
        ))

    async def _handle_chat_send(self, req: JsonRpcRequest, params: dict):
        """Send a user message to RumiLive."""
        content = params.get("content", "")
        if not content:
            send_message(JsonRpcResponse(
                id=req.id, error={"code": -32602, "message": "Missing content"},
            ))
            return

        send_message(JsonRpcResponse(id=req.id, result={"status": "queued"}))

        if self._rumi:
            self._loop.call_soon_threadsafe(
                self._rumi._on_text_command, content
            )

    async def _handle_chat_interrupt(self, req: JsonRpcRequest):
        """Interrupt the current turn."""
        if self._rumi and self._rumi.ui:
            self._rumi.ui._interrupt_requested.set()
        send_message(JsonRpcResponse(id=req.id, result={"status": "interrupted"}))

    async def _handle_slash(self, req: JsonRpcRequest, params: dict):
        """Execute a slash command."""
        command = params.get("command", "")
        args = params.get("args", "")
        full = f"/{command} {args}".strip() if args else f"/{command}"

        send_message(JsonRpcResponse(id=req.id, result={"status": "executing"}))

        if self._rumi:
            self._loop.call_soon_threadsafe(
                self._rumi.ui.handle_command, full
            )

    def emit(self, method: str, params: dict):
        """Thread-safe method to emit an event to the frontend."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(
                send_message, JsonRpcEvent(method=method, params=params)
            )
        else:
            send_message(JsonRpcEvent(method=method, params=params))


class GatewayAdapter:
    """Drop-in replacement for RumiUI that emits JSON-RPC events."""

    def __init__(self, server: GatewayServer):
        self._server = server
        self._running = True
        self._think_mode = False
        self._deep_dive_active = False
        self._rumi_state = "READY"
        self.status_text = "READY"
        self._start_time = time.time()
        self._personality = "professional"
        self._discovery_running = False
        self._discovery_step = ""
        self._discovery_topic = ""
        self._total_tokens = 0
        self._total_cost = 0.0
        self._last_latency = 0.0
        self._message_count = 0
        self._interrupt_requested = threading.Event()
        self._is_busy = False

        self.on_text_command = None
        self.on_discovery_command = None
        self.on_think_mode_toggle = None
        self.on_deep_dive_toggle = None
        self.on_idle_scan = None
        self._message_queue_count = 0

        self._timeline = _Timeline()
        self._pipeline = _PipelineTracker()
        self._tool_calls = _ToolCalls()
        self._graph_metrics = _GraphMetrics()
        self._activity_feed = _ActivityFeed()

    def handle_command(self, cmd: str):
        """Dispatch slash commands."""
        cmd_lower = cmd.lower().strip()

        if cmd_lower == "/help":
            self._server.emit("system.message", {
                "content": (
                    "Commands:\n"
                    "  /discover <topic>    Run discovery pipeline\n"
                    "  /cause <observation>  Cause mode (Newton Step)\n"
                    "  /think               Toggle think mode\n"
                    "  /dive                Toggle deep dive mode\n"
                    "  /status              Show session status\n"
                    "  /clear               Clear transcript\n"
                    "  /quit                Exit\n"
                    "\n"
                    "  Ctrl+K  Command palette\n"
                    "  Ctrl+L  Clear screen\n"
                    "  Ctrl+X  Session switcher\n"
                    "  ?       Help overlay\n"
                    "  Esc     Interrupt"
                ),
            })
        elif cmd_lower == "/think":
            self._think_mode = not self._think_mode
            state = "ON" if self._think_mode else "OFF"
            self._server.emit("state.update", {
                "state": f"THINK {state}", "think_mode": self._think_mode,
            })
            self._server.emit("system.message", {"content": f"Think mode: {state}"})
        elif cmd_lower == "/dive":
            self._deep_dive_active = not self._deep_dive_active
            state = "ON" if self._deep_dive_active else "OFF"
            self._server.emit("state.update", {
                "state": f"DIVE {state}", "deep_dive": self._deep_dive_active,
            })
            self._server.emit("system.message", {"content": f"Deep dive: {state}"})
        elif cmd_lower == "/clear":
            if self.on_text_command:
                pass  # handled by frontend
        elif cmd_lower.startswith("/discover "):
            topic = cmd[len("/discover "):].strip()
            if topic:
                self.set_discovery_topic(topic)
                self._server.emit("system.message", {"content": f"Starting discovery: {topic}"})
                if self.on_discovery_command:
                    threading.Thread(
                        target=self.on_discovery_command,
                        args=("discover", topic), daemon=True,
                    ).start()
            else:
                self._server.emit("system.message", {"content": "Usage: /discover <topic>"})
        elif cmd_lower.startswith("/cause "):
            observation = cmd[len("/cause "):].strip()
            if observation:
                self._server.emit("system.message", {"content": f"Cause mode: {observation}"})
                if self.on_discovery_command:
                    threading.Thread(
                        target=self.on_discovery_command,
                        args=("cause", observation), daemon=True,
                    ).start()
            else:
                self._server.emit("system.message", {"content": "Usage: /cause <observation>"})
        elif cmd_lower.startswith("/grounded "):
            topic = cmd[len("/grounded "):].strip()
            if topic:
                self.set_discovery_topic(topic)
                if self.on_discovery_command:
                    threading.Thread(
                        target=self.on_discovery_command,
                        args=("grounded", topic), daemon=True,
                    ).start()
        elif cmd_lower == "/status":
            self._server.emit("status.info", {
                "state": self._rumi_state,
                "tokens": self._total_tokens,
                "cost": self._total_cost,
            })
        elif cmd_lower == "/quit":
            self._server.emit("session.ending", {})
        else:
            self._server.emit("system.message", {
                "content": f"Unknown command: {cmd_lower}. Type /help for commands.",
            })

    def _parse_discovery_line(self, text: str):
        """Parse discovery pipeline output to update metrics in real-time."""
        import re
        tl = text.lower().strip()

        # Phase detection: [Phase 1/16] LITERATURE, [Phase 2/16] KNOWLEDGE GRAPH, etc.
        phase_match = re.search(r'\[phase\s+([\d.]+)/\d+\]\s*(.+?)(?:\s*[—\-]|$)', tl)
        if phase_match:
            phase_name = phase_match.group(2).strip()
            self._pipeline.start_phase(phase_name)
            self._discovery_step = phase_name
            pct = self._pipeline.get_progress_pct()
            self._server.emit("discovery.phase", {
                "phase": phase_name, "progress": pct,
                "topic": self._discovery_topic,
            })

        # Entity/relationship extraction from graph output
        ent_match = re.search(r'(\d+)\s*entities?,\s*(\d+)\s*relationships?', tl)
        if ent_match:
            self._graph_metrics.update(
                entities=int(ent_match.group(1)),
                relationships=int(ent_match.group(2)),
                nodes=int(ent_match.group(1)),
                edges=int(ent_match.group(2)),
            )
            self._server.emit("graph.update", self._graph_metrics.snapshot())

        # Paper count
        paper_match = re.search(r'(\d+)\s*papers?\b', tl)
        if paper_match and ("literature" in tl or "total" in tl or "round" in tl or "found" in tl):
            self._graph_metrics.update(papers=int(paper_match.group(1)))

        # Cluster/component count
        cluster_match = re.search(r'components?:\s*(\d+)', tl)
        if cluster_match:
            self._graph_metrics.update(clusters=int(cluster_match.group(1)))

        # Anomalies
        anom_match = re.search(r'found\s+(\d+)\s+anomal', tl)
        if anom_match:
            self._graph_metrics.update(novelty_candidates=int(anom_match.group(1)))

        # Contradictions
        contra_match = re.search(r'(\d+)\s+contradictions?', tl)
        if contra_match:
            self._graph_metrics.update(contradictions=int(contra_match.group(1)))

        # Score output
        score_match = re.search(r'score:\s*([\d.]+)/100', tl)
        if score_match:
            self._activity_feed.add("score", f"{score_match.group(1)}/100")

    def write_log(self, text: str):
        """Intercept output and emit as JSON-RPC events."""
        import re
        tl = text.lower().strip()

        if any(skip in tl for skip in ["[luka]", "[episodic]", "[selfawareness]",
                                        "[coordinator]", "[dreaming]", "[vectormemory]",
                                        "[telegram]", "non-data parts", "[focus]"]):
            return

        # Parse discovery pipeline metrics from output
        if self._discovery_running:
            self._parse_discovery_line(text)

        if tl.startswith("you:"):
            content = text[4:].strip()
            self._server.emit("user.message", {"content": content})
        elif tl.startswith("luka:") or tl.startswith("ai:"):
            prefix_len = 5 if tl.startswith("luka:") else 3
            content = text[prefix_len:].strip()
            content = re.sub(r'\*\*Thinking(?:\s*\(.*?\))?\*\*\s*', '', content).strip()
            content = re.sub(r'\*\*Reasoning\*\*\s*', '', content).strip()
            if content:
                self._server.emit("assistant.message", {"content": content})
        elif tl.startswith("sys:"):
            content = text[4:].strip()
            self._server.emit("system.message", {"content": content})
        elif tl.startswith("err:") or tl.startswith("error:"):
            err_msg = text[4:].strip()
            self._server.emit("error", {"message": err_msg})
        elif tl.startswith("sec:"):
            sec_msg = text[4:].strip()
            self._server.emit("system.message", {"content": f"⚠ {sec_msg}"})
        else:
            clean = re.sub(r'\[/?\w+(?: \w+=[^\]]+)*\]', '', text).strip()
            if clean:
                self._server.emit("assistant.message", {"content": clean})

    def set_state(self, state: str):
        self._rumi_state = state
        self.status_text = state
        self._is_busy = state in ("THINKING", "PROCESSING", "SPEAKING")
        self._server.emit("state.update", {"state": state})

    def set_discovery_topic(self, topic: str):
        """Set the current discovery topic (called before pipeline starts)."""
        self._discovery_topic = topic
        self._discovery_running = True
        self._pipeline.reset()
        self._pipeline._topic = topic
        self._pipeline._running = True
        self._timeline.add("discovery.start", topic)

    def set_discovery_step(self, step: str):
        self._discovery_step = step
        self._discovery_running = True
        self._pipeline.start_phase(step)
        self._timeline.add("discovery.phase", step)
        self._activity_feed.add("pipeline", step)
        pct = self._pipeline.get_progress_pct()
        self._server.emit("discovery.phase", {
            "phase": step, "progress": pct, "topic": self._discovery_topic,
        })

    def set_discovery_done(self, report: dict = None):
        """Mark discovery complete. If report provided, extract real metrics."""
        if report:
            self._graph_metrics.update_from_report(report)
        self._pipeline._running = False
        self._discovery_running = False
        self._discovery_step = ""
        self._timeline.add("discovery.done", "pipeline complete")
        self._activity_feed.add("discovery.done", f"{self._graph_metrics.papers} papers, {self._graph_metrics.entities} entities")
        self._server.emit("discovery.done", {
            "papers": self._graph_metrics.papers,
            "entities": self._graph_metrics.entities,
            "edges": self._graph_metrics.edges,
        })

    def show_thinking(self, message: str = ""):
        self._server.emit("thinking.start", {"verb": message or "thinking"})

    def show_done(self, message: str = ""):
        self._server.emit("thinking.done", {})

    def show_tool_call(self, name: str, query: str = ""):
        self._tool_calls.start(name, query)
        self._activity_feed.add("tool", name)
        self._server.emit("tool.start", {"name": name, "query": query})

    def complete_tool_call(self, name: str, result: str = ""):
        self._tool_calls.complete(name, result)
        self._server.emit("tool.complete", {"name": name, "result": result})

    def update_tokens(self, tokens: int, cost: float = 0.0, latency: float = 0.0):
        self._total_tokens += tokens
        self._total_cost += cost
        if latency > 0:
            self._last_latency = latency
        self._server.emit("metrics.update", {
            "tokens": self._total_tokens,
            "cost": self._total_cost,
            "latency": latency,
        })

    def update_graph_metrics(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self._graph_metrics, k):
                setattr(self._graph_metrics, k, v)
        snap = self._graph_metrics.snapshot()
        self._server.emit("graph.update", snap)

    def show_toast(self, message: str, duration: float = 2.5):
        self._server.emit("system.message", {"content": message})

    def wait_for_api_key(self):
        pass

    def interrupt_requested(self):
        return self._interrupt_requested.is_set()

    def clear_interrupt(self):
        self._interrupt_requested.clear()

    def feed_amplitude(self, amplitude: float):
        pass


class _Timeline:
    """Real timeline that tracks pipeline events."""
    def __init__(self):
        self._events: list[dict] = []

    def add(self, event_type: str, detail: str = "", **kw):
        self._events.append({
            "type": event_type,
            "detail": detail,
            "time": time.time(),
            **kw,
        })

    def get_events(self, last_n=20):
        return self._events[-last_n:]

    def clear(self):
        self._events.clear()

    def count(self):
        return len(self._events)


class _PipelineTracker:
    """Tracks discovery pipeline phase progress with real percentages."""
    # Phase name → (order index, total phases)
    PHASE_MAP = {
        "literature":               0,
        "citation network":         1,
        "knowledge graph":          2,
        "enrichment":               3,
        "graph inference":          4,
        "gap detection":            5,
        "gap":                      5,
        "anomaly detection":        6,
        "anomaly":                  6,
        "hidden variables":         7,
        "missing variables":        7,
        "mechanism":                8,
        "mechanism generation":     8,
        "derivation":               9,
        "math chain":               9,
        "completeness":            10,
        "prediction":              11,
        "observability":           12,
        "theory":                  13,
        "theory competition":      13,
        "adversarial":             14,
        "peer review":             15,
        "computational":           16,
        "experiment":              17,
        "data analysis":           18,
        "contradictions":          19,
        "skeptic":                 20,
        "cross-validation":        21,
        "scoring":                 22,
        "discovery scoring":       22,
        "refinement":              23,
        "reflexion":               24,
        "self-improvement":        25,
        "gap discovery":           26,
        # Cause mode / Newton step
        "curious questioning":      -1,
        "newton step":             -1,
        "running v2":               0,
        "refining":                23,
        "reflexion:":              24,
    }
    TOTAL_PHASES = 27

    def __init__(self):
        self.reset()

    def reset(self):
        self._current_phase = ""
        self._current_index = -1
        self._completed: list[str] = []
        self._topic = ""
        self._running = False

    def start_phase(self, phase_name: str):
        self._running = True
        self._current_phase = phase_name
        idx = self._resolve_index(phase_name)
        if idx >= 0:
            self._current_index = idx

    def complete_phase(self, phase_name: str):
        if phase_name not in self._completed:
            self._completed.append(phase_name)

    def error_phase(self, phase_name: str):
        pass  # still counts as attempted

    def all_done(self):
        return not self._running

    def get_display(self):
        return self._completed

    def _resolve_index(self, phase_name: str) -> int:
        lower = phase_name.lower().strip()
        for key, idx in self.PHASE_MAP.items():
            if key in lower:
                return idx
        return max(self._current_index, 0)

    def get_progress_pct(self) -> float:
        if not self._running:
            return 0
        idx = self._current_index if self._current_index >= 0 else 0
        return min(95, int((idx / self.TOTAL_PHASES) * 100))


class _ToolCalls:
    """Real tool call tracker."""
    def __init__(self):
        self._calls: list[dict] = []

    def start(self, name: str, query: str = "", **kw):
        self._calls.append({
            "name": name, "query": query, "status": "running",
            "start": time.time(), **kw,
        })

    def complete(self, name: str, result: str = "", **kw):
        for c in reversed(self._calls):
            if c["name"] == name and c["status"] == "running":
                c["status"] = "done"
                c["result"] = result
                c["elapsed"] = time.time() - c["start"]
                break

    def error(self, name: str, **kw):
        for c in reversed(self._calls):
            if c["name"] == name and c["status"] == "running":
                c["status"] = "error"
                break

    def get_recent(self, n=10):
        return self._calls[-n:]


class _GraphMetrics:
    """Real graph metrics that update from discovery pipeline output."""
    def __init__(self):
        self.nodes = 0
        self.edges = 0
        self.clusters = 0
        self.contradictions = 0
        self.novelty_candidates = 0
        self.papers = 0
        self.entities = 0
        self.relationships = 0

    def update(self, **kw):
        for k, v in kw.items():
            if hasattr(self, k) and isinstance(v, (int, float)):
                setattr(self, k, v)

    def update_from_report(self, report: dict):
        """Extract real metrics from a discovery report dict."""
        phases = report.get("phases", {})
        self.papers = phases.get("literature", {}).get("papers_found", self.papers)
        kg = phases.get("knowledge_graph", {})
        self.entities = kg.get("entities", self.entities)
        self.relationships = kg.get("relationships", self.relationships)
        self.nodes = self.entities
        self.edges = self.relationships
        self.clusters = kg.get("components", self.clusters)
        anomalies = phases.get("anomaly_detection", {})
        self.novelty_candidates = anomalies.get("total_anomalies", self.novelty_candidates)
        contradictions = phases.get("contradictions", {})
        self.contradictions = contradictions.get("total_contradictions", self.contradictions)

    def snapshot(self):
        return {k: getattr(self, k) for k in [
            "nodes", "edges", "clusters", "contradictions",
            "novelty_candidates", "papers", "entities", "relationships",
        ]}


class _ActivityFeed:
    """Real activity feed for tracking pipeline events."""
    def __init__(self):
        self._items: list[dict] = []

    def add(self, activity_type: str, detail: str = "", **kw):
        self._items.append({
            "type": activity_type,
            "detail": detail,
            "time": time.time(),
            **kw,
        })
        # Keep last 100
        if len(self._items) > 100:
            self._items = self._items[-100:]

    def get_recent(self, n=10):
        return self._items[-n:]

    def clear(self):
        self._items.clear()
