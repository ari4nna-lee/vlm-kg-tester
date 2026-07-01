"""
scene_reasoner.py
=================
Type 2 reasoning pass over the accumulated pipeline outputs.

Runs AFTER pipeline.py completes (post-processing mode for now).
Takes the global KG, priority map, mosaic, and waypoints produced
by the pipeline and runs a slower, deliberate LLM reasoning pass
that produces:
  - A high-level scene narrative ("whole picture" understanding)
  - Priority adjustments for specific KG nodes/regions
  - Refined search directives with reasoning
  - Temporal anomaly flags (nodes that appeared and disappeared)
  - A queryable chat interface over the scene

Typical usage (post-processing):
    python scene_reasoner.py
    # reads from ./results/, writes to ./results/scene_reasoning/

Feed-back hook (future pipeline integration):
    reasoner = SceneReasoner(cfg)
    adjustments = reasoner.run(kg_graph, mosaic_path, priority_map_path, waypoints)
    # apply adjustments.priority_overrides back into self._kg_graph
"""

from __future__ import annotations

import json
import logging
import time
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import dataclasses

import numpy as np
import cv2
import networkx as nx
from networkx.readwrite import json_graph

import os
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

RESULTS_DIR = Path("./results")
JSON_DIR = RESULTS_DIR / "json"
OUTPUT_DIR = RESULTS_DIR / "scene_reasoning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SceneReasonerConfig:
    backend: Literal["gemini", "vllm"] = "gemini"

    gemini_api_key: str = os.getenv("GEMINI_API_KEY")         
    gemini_model: str = "gemini-2.5-flash"

    # vLLM local (swap backend to "vllm" and set these)
    grpc_endpoint: str = "http://localhost:8000/v1"
    vllm_model: str = "google/gemma-4-E2B-it"

    # Reasoning params
    max_tokens: int = 4096 
    temperature: float = 0.3          # slightly higher than VLM — want creative reasoning

    # KG compression params
    top_n_nodes: int = 25             # max nodes sent to LLM
    top_n_edges: int = 40             # max edges sent to LLM
    min_priority_for_summary: float = 0.3   # filter out very low priority nodes

    # Task context
    task_prompt: str = "Identify vehicles and areas where you would most likely find a truck"


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PriorityOverride:
    """
    A reasoned adjustment to a specific KG node's priority.
    Applied back into the graph by the pipeline (future) or post-hoc.
    """
    node_id: str
    new_priority: float
    reason: str


@dataclass
class SearchDirective:
    """
    A refined search target produced by Type 2 reasoning.
    May differ from search_planner.py waypoints because it incorporates
    reasoning about relationships, not just spatial clustering.
    """
    global_x: float
    global_y: float
    priority: float
    reasoning: str                    # natural language explanation
    supporting_nodes: list[str]       # node_ids that justify this directive


@dataclass
class TemporalAnomaly:
    """
    A node that exhibited significant change across observation history —
    appeared, disappeared, or had high priority variance.
    Worth revisiting with the drone.
    """
    node_id: str
    label: str
    anomaly_type: str                 # "disappeared", "priority_spike", "priority_drop"
    description: str


@dataclass
class SceneReasoningResult:
    """
    Full output of one SceneReasoner.run() pass.
    Written to disk and optionally fed back into the pipeline.
    """
    timestamp: float
    scene_narrative: str              # high-level whole-scene understanding
    priority_overrides: list[PriorityOverride]
    search_directives: list[SearchDirective]
    temporal_anomalies: list[TemporalAnomaly]
    raw_llm_response: str             # for debugging/audit
    compressed_kg_summary: dict       # the compressed KG that was sent to the LLM


# ---------------------------------------------------------------------------
# KG Compression
# ---------------------------------------------------------------------------

class KGCompressor:
    """
    Compresses the global NetworkX KG into an LLM-digestible summary.

    The full graph at scale has hundreds of nodes and thousands of edges —
    far too large for an LLM context window and semantically redundant
    (many low-priority background nodes add noise, not signal).

    Compression strategy:
    1. Filter to top-N nodes by priority
    2. Include all edges between those nodes
    3. Summarize observation history to key stats (not raw list)
    4. Group nodes by semantic class for structural overview
    5. Extract cluster summaries from high-density regions
    """

    def __init__(self, cfg: SceneReasonerConfig):
        self.cfg = cfg

    def compress(self, G: nx.DiGraph) -> dict:

        if not isinstance(G, nx.DiGraph):
            raise TypeError(f"Expected nx.DiGraph, got {type(G)}")
        """
        Returns a dict structured for LLM consumption — compact but information-rich.
        """
        if G.number_of_nodes() == 0:
            return {"error": "empty graph", "nodes": [], "edges": [], "clusters": []}

        # --- step 1: score and filter nodes ---
        scored = []
        for nid, data in G.nodes(data=True):
            priority = data.get("priority", 0.0)
            confidence = data.get("confidence", 0.0)
            obs_history = data.get("observation_history", [])
            n_obs = len(obs_history) if isinstance(obs_history, list) else 0
            # composite score: priority weighted by confidence and observation count
            score = priority * (0.7 + 0.3 * min(confidence, 1.0))
            scored.append((nid, data, score, n_obs))

        scored.sort(key=lambda x: x[2], reverse=True)
        top_nodes = scored[:self.cfg.top_n_nodes]
        top_node_ids = {x[0] for x in top_nodes}

        # --- step 2: build compressed node list ---
        compressed_nodes = []
        for nid, data, score, n_obs in top_nodes:
            if score < self.cfg.min_priority_for_summary:
                continue

            gp = data.get("global_pose", {})
            global_centroid = gp.get("centroid", None) if gp else None

            # summarize observation history instead of dumping raw list
            obs_summary = self._summarize_observations(data.get("observation_history", []))

            compressed_nodes.append({
                "node_id": nid,
                "label": data.get("label", "?"),
                "semantic_class": data.get("semantic_class", "unknown"),
                "priority": round(float(data.get("priority", 0.0)), 3),
                "confidence": round(float(data.get("confidence", 0.0)), 3),
                "observation_count": n_obs,
                "global_centroid": (
                    [round(global_centroid[0], 1), round(global_centroid[1], 1)]
                    if global_centroid else None
                ),
                "mask_area_pct": round(float(data.get("mask_area", 0.0)) * 100, 2),
                "observation_summary": obs_summary,
            })

        # --- step 3: edges between top nodes only ---
        compressed_edges = []
        for u, v, edata in G.edges(data=True):
            if u not in top_node_ids or v not in top_node_ids:
                continue
            compressed_edges.append({
                "from": u,
                "to": v,
                "predicate": edata.get("predicate", "?"),
                "confidence": round(float(edata.get("confidence", 0.0)), 3),
                "tier": edata.get("tier", 0),
            })
        # limit edge count
        compressed_edges = sorted(
            compressed_edges, key=lambda e: e["confidence"], reverse=True
        )[:self.cfg.top_n_edges]

        # --- step 4: class-level structural overview ---
        class_summary = {}
        for nid, data in G.nodes(data=True):
            sc = data.get("semantic_class", "unknown")
            p = data.get("priority", 0.0)
            if sc not in class_summary:
                class_summary[sc] = {"count": 0, "avg_priority": 0.0, "max_priority": 0.0}
            class_summary[sc]["count"] += 1
            class_summary[sc]["avg_priority"] += p
            class_summary[sc]["max_priority"] = max(class_summary[sc]["max_priority"], p)

        for sc in class_summary:
            n = class_summary[sc]["count"]
            class_summary[sc]["avg_priority"] = round(class_summary[sc]["avg_priority"] / n, 3)
            class_summary[sc]["max_priority"] = round(class_summary[sc]["max_priority"], 3)

        # --- step 5: temporal anomalies (computed here, surfaced to LLM) ---
        anomalies = self._detect_temporal_anomalies(G)

        return {
            "graph_stats": {
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "nodes_shown": len(compressed_nodes),
                "edges_shown": len(compressed_edges),
            },
            "class_overview": class_summary,
            "top_nodes": compressed_nodes,
            "top_edges": compressed_edges,
            "temporal_anomalies": anomalies,
        }

    @staticmethod
    def _summarize_observations(obs_history: list) -> dict:
        """Convert raw observation list to key stats for LLM."""
        if not obs_history or not isinstance(obs_history, list):
            return {"count": 0}

        # obs_history entries may be NodeObservation dataclasses (serialized
        # to dicts by the JSON encoder) or raw dicts from the graph
        priorities = []
        frame_ids = []
        for obs in obs_history:
            if isinstance(obs, dict):
                p = obs.get("priority")
                f = obs.get("frame_id")
            elif dataclasses.is_dataclass(obs):
                p = getattr(obs, "priority", None)
                f = getattr(obs, "frame_id", None)
            else:
                continue
            if p is not None:
                priorities.append(float(p))
            if f is not None:
                frame_ids.append(int(f))

        if not priorities:
            return {"count": len(obs_history)}

        return {
            "count": len(priorities),
            "first_seen_frame": min(frame_ids) if frame_ids else None,
            "last_seen_frame": max(frame_ids) if frame_ids else None,
            "priority_mean": round(float(np.mean(priorities)), 3),
            "priority_max": round(float(max(priorities)), 3),
            "priority_min": round(float(min(priorities)), 3),
            "priority_trend": "increasing" if len(priorities) > 1 and priorities[-1] > priorities[0] else
                              "decreasing" if len(priorities) > 1 and priorities[-1] < priorities[0] else
                              "stable",
        }

    @staticmethod
    def _detect_temporal_anomalies(G: nx.DiGraph) -> list[dict]:
        """
        Flag nodes with unusual observation patterns.
        Surfaced in the compressed KG so the LLM can reason about them.
        """
        anomalies = []
        for nid, data in G.nodes(data=True):
            obs = data.get("observation_history", [])
            if not isinstance(obs, list) or len(obs) < 2:
                continue

            priorities = []
            for o in obs:
                if isinstance(o, dict):
                    p = o.get("priority")
                elif dataclasses.is_dataclass(o):
                    p = getattr(o, "priority", None)
                else:
                    continue
                if p is not None:
                    priorities.append(float(p))

            if len(priorities) < 2:
                continue

            variance = float(np.var(priorities))
            trend = priorities[-1] - priorities[0]

            if variance > 0.05:
                anomalies.append({
                    "node_id": nid,
                    "label": data.get("label", "?"),
                    "semantic_class": data.get("semantic_class", "unknown"),
                    "anomaly_type": "high_priority_variance",
                    "variance": round(variance, 4),
                    "priority_range": [round(min(priorities), 3), round(max(priorities), 3)],
                })
            elif trend < -0.3:
                anomalies.append({
                    "node_id": nid,
                    "label": data.get("label", "?"),
                    "semantic_class": data.get("semantic_class", "unknown"),
                    "anomaly_type": "priority_drop",
                    "trend": round(trend, 3),
                })

        return anomalies[:10]   # cap at 10 — don't overwhelm the LLM


# ---------------------------------------------------------------------------
# LLM Backends
# ---------------------------------------------------------------------------

def _encode_image_b64(image_path: Path) -> Optional[str]:
    """Load an image from disk and base64-encode it for the API."""
    if not image_path.exists():
        log.warning("Image not found: %s", image_path)
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_narrative": {"type": "string"},
        "key_findings": {"type": "array", "items": {"type": "string"}},
        "search_strategy": {"type": "string"},
        "priority_overrides": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "new_priority": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["node_id", "new_priority"],
            },
        },
        "search_directives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "global_x": {"type": "number"},
                    "global_y": {"type": "number"},
                    "priority": {"type": "number"},
                    "reasoning": {"type": "string"},
                    "supporting_nodes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["global_x", "global_y", "priority"],
            },
        },
        "temporal_anomalies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "label": {"type": "string"},
                    "anomaly_type": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
    },
    "required": ["scene_narrative", "priority_overrides", "search_directives", "temporal_anomalies"],
}

def _build_gemini_request(
    kg_summary: dict,
    task_prompt: str,
    mosaic_b64: Optional[str],
    priority_map_b64: Optional[str],
):
    """
    Gemini-native request format (NO OpenAI message structure).
    """

    system_instruction = f"""
You are a high-level scene understanding module for an autonomous multi-drone system.

Task: {task_prompt}

You will receive:
- A compressed knowledge graph
- Optional aerial mosaic image
- Optional priority heatmap

Return ONLY valid JSON with:
scene_narrative, key_findings, priority_overrides,
search_directives, temporal_anomalies, search_strategy
""".strip()

    contents = []

    # ---- KG TEXT (always first) ----
    contents.append(
        f"Knowledge Graph Summary:\n{json.dumps(kg_summary, indent=2)}"
    )

    # ---- IMAGE: mosaic ----
    if mosaic_b64:
        contents.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": mosaic_b64
            }
        })
        contents.append("Above: global mosaic image")

    # ---- IMAGE: priority map ----
    if priority_map_b64:
        contents.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": priority_map_b64
            }
        })
        contents.append("Above: priority heatmap")

    return system_instruction, contents


def _call_gemini(question: str, kg_graph: nx.DiGraph, cfg: SceneReasonerConfig) -> str:
    import json
    from google import genai

    compressor = KGCompressor(cfg)
    kg_summary = compressor.compress(kg_graph)

    system_instruction = f"""
You are a scene intelligence assistant for a drone system.

Task: {cfg.task_prompt}

Answer using ONLY the provided knowledge graph.
Return concise reasoning and reference node IDs if possible.
""".strip()

    contents = [
        f"Knowledge Graph:\n{json.dumps(kg_summary, indent=2)}",
        f"Question: {question}",
    ]

    client = genai.Client(api_key=cfg.gemini_api_key)

    response = client.models.generate_content(
        model=cfg.gemini_model,
        contents=contents,
        config={
            "system_instruction": system_instruction,
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_tokens,
            "response_mime_type": "application/json",
        },
    )

    return response.text

def _call_gemini_raw(system_instruction: str, contents: list, cfg: SceneReasonerConfig) -> str:
    from google import genai

    client = genai.Client(api_key=cfg.gemini_api_key)
    response = client.models.generate_content(
        model=cfg.gemini_model,
        contents=contents,
        config={
            "system_instruction": system_instruction,
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_tokens,
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_SCHEMA,
        },
    )
    return response.text


_BACKENDS = {
    "gemini": _call_gemini,
}

# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    import re
    text = raw.strip()
    # strip markdown fences if present
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # find outermost JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1:
        log.warning("scene_reasoner: no JSON found in LLM response")
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as e:
        log.warning("scene_reasoner: JSON parse failed: %s", e)
        return {}

def _safe_float(value, default: float = 0.0) -> float:
    """Coerce a value to float, treating None/missing/unparsable as default."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SceneReasoner:
    """
    Type 2 reasoning pass over the accumulated pipeline outputs.

    Post-processing mode (current):
        reasoner = SceneReasoner(cfg)
        result = reasoner.run_from_disk()

    Pipeline feed-back mode (future):
        result = reasoner.run(kg_graph, mosaic_path, priority_map_path, waypoints)
        # apply result.priority_overrides back into pipeline._kg_graph
    """

    def __init__(self, cfg: Optional[SceneReasonerConfig] = None):
        self.cfg = cfg or SceneReasonerConfig()
        self.compressor = KGCompressor(self.cfg)
        if self.cfg.backend not in _BACKENDS:
            raise ValueError(f"Unknown backend '{self.cfg.backend}'. Choose: {list(_BACKENDS)}")
        log.info("SceneReasoner initialized — backend=%s", self.cfg.backend)

    # ------------------------------------------------------------------
    # Post-processing entry point (runs from disk after pipeline.py)
    # ------------------------------------------------------------------

    def run_from_disk(self) -> SceneReasoningResult:
        """
        Load all pipeline outputs from disk and run the reasoning pass.
        Called as a standalone script after pipeline.py completes.
        """
        log.info("scene_reasoner: loading pipeline outputs from %s", RESULTS_DIR)

        # load global KG by reconstructing from per-frame JSONs
        kg_graph = self._load_kg_from_json_dir()

        mosaic_path = RESULTS_DIR / "global_mosaic.png"
        priority_map_path = RESULTS_DIR / "global_priority_map.png"

        # load waypoints if they exist
        waypoints_path = OUTPUT_DIR / "waypoints.json"
        waypoints = []
        if waypoints_path.exists():
            with open(waypoints_path) as f:
                waypoints = json.load(f)

        return self.run(
            kg_graph=kg_graph,
            mosaic_path=mosaic_path,
            priority_map_path=priority_map_path,
            waypoints=waypoints,
        )

    def _load_kg_from_json_dir(self) -> nx.DiGraph:
        """
        Reconstruct a global graph from per-frame JSON outputs.
        Uses node_link_data format written by output_worker.
        Simple merge by node id — this is the post-hoc version,
        not the full spatial_merge the live pipeline uses.
        """
        G = nx.DiGraph()
        frame_jsons = sorted(JSON_DIR.glob("frame_*.json"))
        if not frame_jsons:
            log.warning("scene_reasoner: no frame JSONs found in %s", JSON_DIR)
            return G

        log.info("scene_reasoner: loading %d frame JSONs", len(frame_jsons))
        for path in frame_jsons:
            try:
                with open(path) as f:
                    data = json.load(f)
                frame_graph_data = data.get("graph", {})
                if not frame_graph_data:
                    continue
                frame_G = nx.DiGraph()

                for node in frame_graph_data.get("nodes", []):
                    nid = node.get("id")
                    if nid is None:
                        continue
                    frame_G.add_node(nid, **node)

                for edge in frame_graph_data.get("links", []):
                    u = edge.get("source")
                    v = edge.get("target")
                    if u is None or v is None:
                        continue
                    frame_G.add_edge(u, v, **edge)
                    
                for nid, ndata in frame_G.nodes(data=True):
                    if nid not in G:
                        G.add_node(nid, **ndata)
                    else:
                        # update priority if this frame's version is higher
                        existing_p = G.nodes[nid].get("priority", 0.0)
                        new_p = ndata.get("priority", 0.0)
                        if new_p > existing_p:
                            G.nodes[nid].update(ndata)
                for u, v, edata in frame_G.edges(data=True):
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, **edata)
            except Exception as e:
                log.warning("scene_reasoner: failed to load %s: %s", path.name, e)

        log.info("scene_reasoner: reconstructed graph — %d nodes, %d edges",
                 G.number_of_nodes(), G.number_of_edges())
        return G

    # ------------------------------------------------------------------
    # Core reasoning pass
    # (this signature is the future pipeline feed-back interface)
    # ------------------------------------------------------------------

    def run(
        self,
        kg_graph: nx.DiGraph,
        mosaic_path: Optional[Path] = None,
        priority_map_path: Optional[Path] = None,
        waypoints: Optional[list] = None,
    ) -> SceneReasoningResult:
        """
        Run one Type 2 reasoning pass.

        Parameters
        ----------
        kg_graph : nx.DiGraph
            The global KG — either reconstructed from disk (post-processing)
            or pipeline._kg_graph directly (future live mode).
        mosaic_path : Path, optional
            Path to global_mosaic.png
        priority_map_path : Path, optional
            Path to global_priority_map.png
        waypoints : list, optional
            Output of extract_search_waypoints(), serialized to dicts

        Returns
        -------
        SceneReasoningResult
            Full reasoning output — written to disk and returned for
            future pipeline feed-back.
        """
        t0 = time.perf_counter()

        # --- compress KG ---
        log.info("scene_reasoner: compressing KG (%d nodes)...", kg_graph.number_of_nodes())
        kg_summary = self.compressor.compress(kg_graph)

        # add waypoints to summary if available
        if waypoints:
            kg_summary["current_waypoints"] = waypoints[:5]

        # --- load images ---
        mosaic_b64 = _encode_image_b64(mosaic_path) if mosaic_path else None
        priority_map_b64 = _encode_image_b64(priority_map_path) if priority_map_path else None

        log.info("scene_reasoner: mosaic=%s, priority_map=%s",
                 "loaded" if mosaic_b64 else "not found",
                 "loaded" if priority_map_b64 else "not found")

        # --- build messages and call LLM ---
        system_instruction, contents = _build_gemini_request(
            kg_summary=kg_summary,
            task_prompt=self.cfg.task_prompt,
            mosaic_b64=mosaic_b64,
            priority_map_b64=priority_map_b64,
        )

        log.info("scene_reasoner: calling %s backend...", self.cfg.backend)
        raw_response = _call_gemini_raw(system_instruction, contents, self.cfg)
        elapsed = time.perf_counter() - t0
        log.info("scene_reasoner: LLM response in %.2fs", elapsed)

        # --- parse response ---
        parsed = _parse_response(raw_response)

        result = SceneReasoningResult(
            timestamp=time.time(),
            scene_narrative=parsed.get("scene_narrative", ""),
            priority_overrides=[
                PriorityOverride(
                    node_id=p["node_id"],
                    new_priority=_safe_float(p.get("new_priority"), 0.0),
                    reason=p.get("reason", ""),
                )
                for p in parsed.get("priority_overrides", [])
                if isinstance(p, dict) and "node_id" in p and "new_priority" in p
            ],
            search_directives=[
                SearchDirective(
                    global_x=_safe_float(d.get("global_x"), 0.0),
                    global_y=_safe_float(d.get("global_y"), 0.0),
                    priority=_safe_float(d.get("priority"), 0.0),
                    reasoning=d.get("reasoning", ""),
                    supporting_nodes=d.get("supporting_nodes", []),
                )
                for d in parsed.get("search_directives", [])
                if isinstance(d, dict)
            ],
            temporal_anomalies=[
                TemporalAnomaly(
                    node_id=a.get("node_id", ""),
                    label=a.get("label", ""),
                    anomaly_type=a.get("anomaly_type", ""),
                    description=a.get("description", ""),
                )
                for a in parsed.get("temporal_anomalies", [])
            ],
            raw_llm_response=raw_response,
            compressed_kg_summary=kg_summary,
        )

        self._write_outputs(result, parsed)
        return result

    # ------------------------------------------------------------------
    # Chat interface (Task 3 seed)
    # ------------------------------------------------------------------

    def query(self, question: str, kg_graph: nx.DiGraph) -> str:
        """
        Ask a natural language question about the current scene.

        Example:
            reasoner.query("Where is the most likely location of the truck?", G)
            reasoner.query("What roads are traversable in the eastern region?", G)
            reasoner.query("Which nodes have disappeared since they were first seen?", G)

        This is the Task 3 chat interface — queryable KG via natural language.
        """
        log.info("QUERY KG TYPE: %s", type(kg_graph))
        kg_summary = self.compressor.compress(kg_graph)

        messages = [
            {
                "role": "system",
                "content": f"""You are a scene intelligence assistant for a drone search mission.
Task: {self.cfg.task_prompt}
You have access to a compressed knowledge graph of the search area.
Answer questions concisely and specifically. Reference node IDs and locations where relevant."""
            },
            {
                "role": "user",
                "content": f"""Knowledge Graph:\n```json\n{json.dumps(kg_summary, indent=2)}\n```\n\nQuestion: {question}"""
            }
        ]

        log.info("scene_reasoner.query: '%s'", question)

        response = _BACKENDS[self.cfg.backend](
            question=question,
            kg_graph=kg_graph,
            cfg=self.cfg,
        )

        # log the exchange
        qa_log_path = OUTPUT_DIR / "chat_log.jsonl"
        with open(qa_log_path, "a") as f:
            f.write(json.dumps({
                "timestamp": time.time(),
                "question": question,
                "answer": response,
            }) + "\n")

        return response

    # ------------------------------------------------------------------
    # Output writing
    # ------------------------------------------------------------------

    def _write_outputs(self, result: SceneReasoningResult, parsed: dict):
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")

        # --- main reasoning output ---
        out = {
            "timestamp": result.timestamp,
            "scene_narrative": result.scene_narrative,
            "search_strategy": parsed.get("search_strategy", ""),
            "key_findings": parsed.get("key_findings", []),
            "priority_overrides": [dataclasses.asdict(p) for p in result.priority_overrides],
            "search_directives": [dataclasses.asdict(d) for d in result.search_directives],
            "temporal_anomalies": [dataclasses.asdict(a) for a in result.temporal_anomalies],
            "compressed_kg_summary": result.compressed_kg_summary,
            "raw_llm_response": result.raw_llm_response,
        }
        out_path = OUTPUT_DIR / f"reasoning_{timestamp_str}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        log.info("scene_reasoner: written to %s", out_path)

        # --- human-readable summary ---
        summary_path = OUTPUT_DIR / f"summary_{timestamp_str}.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("ARCADIA SCENE REASONING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Scene Narrative:\n{result.scene_narrative}\n\n")
            f.write(f"Search Strategy:\n{parsed.get('search_strategy', 'N/A')}\n\n")
            f.write("Key Findings:\n")
            for i, finding in enumerate(parsed.get("key_findings", []), 1):
                f.write(f"  {i}. {finding}\n")
            f.write(f"\nSearch Directives ({len(result.search_directives)}):\n")
            for i, d in enumerate(result.search_directives, 1):
                f.write(f"  {i}. ({d.global_x:.0f}, {d.global_y:.0f}) "
                        f"priority={d.priority:.3f} — {d.reasoning}\n")
            f.write(f"\nPriority Overrides ({len(result.priority_overrides)}):\n")
            for p in result.priority_overrides:
                f.write(f"  {p.node_id}: → {p.new_priority:.3f} ({p.reason})\n")
            f.write(f"\nTemporal Anomalies ({len(result.temporal_anomalies)}):\n")
            for a in result.temporal_anomalies:
                f.write(f"  [{a.anomaly_type}] {a.label}: {a.description}\n")

        log.info("scene_reasoner: summary written to %s", summary_path)

        # also print narrative to console so it's visible in the terminal
        print("\n" + "=" * 60)
        print("SCENE NARRATIVE:", result.scene_narrative)
        if parsed.get("search_strategy"):
            print("STRATEGY:", parsed["search_strategy"])
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = SceneReasonerConfig(
        backend="gemini",               # swap to "vllm" for local model
        gemini_model="gemini-2.5-flash",
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        # --- vllm alternative ---
        # backend="vllm",
        # vllm_model="google/gemma-4-E2B-it",
        # grpc_endpoint="http://localhost:8000/v1",
        task_prompt="Identify vehicles and areas where you would most likely find a truck",
        top_n_nodes=25,
        top_n_edges=40,
    )

    reasoner = SceneReasoner(cfg)

    # --- post-processing run ---
    result = reasoner.run_from_disk()

    print(f"\nDone. {len(result.search_directives)} search directives, "
          f"{len(result.priority_overrides)} priority overrides, "
          f"{len(result.temporal_anomalies)} temporal anomalies.")
    print(f"Outputs written to {OUTPUT_DIR}/")

    # --- example chat queries ---
    print("\n--- Chat Interface Demo ---")
    from networkx.readwrite import json_graph as jg
    kg = reasoner._load_kg_from_json_dir()

    questions = [
        "Identify vehicles and areas where you would most likely find a truck",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        answer = reasoner.query(q, kg)
        print(f"A: {answer}")