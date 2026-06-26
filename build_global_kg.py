"""
build_global_kg.py
==================
Reads all per-frame JSON files from results/json/ and builds a
full cross-frame knowledge graph, then renders it.

Run after pipeline.py has finished:
    python build_global_kg.py
"""

import json
import math
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from networkx.readwrite import json_graph

JSON_DIR = Path("./results/json")
OUTPUT_DIR = Path("./results")

# ---------------------------------------------------------------------------
# 1. Load all per-frame graphs
# ---------------------------------------------------------------------------

def load_frame_graphs(json_dir: Path) -> list[dict]:
    frames = []
    for path in sorted(json_dir.glob("frame_*.json")):
        with open(path) as f:
            data = json.load(f)
        frames.append(data)
    print(f"Loaded {len(frames)} frames")
    return frames


# ---------------------------------------------------------------------------
# 2. Build global graph
# ---------------------------------------------------------------------------

def build_global_graph(frames: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()

    for frame_data in frames:
        fid = frame_data["frame_id"]
        frame_graph = json_graph.node_link_graph(frame_data["graph"], edges="links")

        # --- add nodes, namespaced by frame so they don't collide ---
        for node_id, attrs in frame_graph.nodes(data=True):
            global_id = f"f{fid}_{node_id}"
            attrs["frame_id"] = fid
            G.add_node(global_id, **attrs)

        # --- add intra-frame edges ---
        for u, v, attrs in frame_graph.edges(data=True):
            G.add_edge(f"f{fid}_{u}", f"f{fid}_{v}", **attrs, edge_type="spatial")

    # --- add cross-frame edges by matching semantic_class + label ---
    _link_across_frames(G, frames)

    return G


def _link_across_frames(G: nx.DiGraph, frames: list[dict]):
    """
    Link nodes across consecutive frames.
    Primary signal: shared track_id (real identity from the tracker).
    Fallback: same semantic_class + label, for nodes where track_id == -1
    (tracking disabled or track lost).
    """
    from collections import defaultdict

    tracked_buckets = defaultdict(list)   # track_id -> [(frame_id, node_id)]
    untracked_buckets = defaultdict(list) # (semantic_class, label) -> [(frame_id, node_id)]

    for node_id, attrs in G.nodes(data=True):
        tid = attrs.get("track_id", -1)
        fid = attrs.get("frame_id", -1)
        if tid is not None and tid != -1:
            tracked_buckets[tid].append((fid, node_id))
        else:
            key = (attrs.get("semantic_class", ""), attrs.get("label", ""))
            untracked_buckets[key].append((fid, node_id))

    # --- primary: link same track_id across consecutive frames ---
    for tid, instances in tracked_buckets.items():
        instances.sort(key=lambda x: x[0])
        for i in range(len(instances) - 1):
            fid_a, node_a = instances[i]
            fid_b, node_b = instances[i + 1]
            if fid_b - fid_a <= 1:
                G.add_edge(node_a, node_b,
                           predicate="same_as",
                           edge_type="temporal",
                           confidence=0.95,   # higher confidence: real tracker match
                           via="track_id")

    # --- fallback: label/class match for untracked nodes only ---
    for key, instances in untracked_buckets.items():
        instances.sort(key=lambda x: x[0])
        for i in range(len(instances) - 1):
            fid_a, node_a = instances[i]
            fid_b, node_b = instances[i + 1]
            if fid_b - fid_a <= 1:
                G.add_edge(node_a, node_b,
                           predicate="same_as",
                           edge_type="temporal",
                           confidence=0.5,    # lower confidence: proxy match only
                           via="label_match")


# ---------------------------------------------------------------------------
# 3. Render
# ---------------------------------------------------------------------------

def render_global_graph(G: nx.DiGraph, output_path: Path):
    fig, ax = plt.subplots(figsize=(20, 14))

    # --- spatial layout using bbox centroid where available ---
    pos = {}
    for node, data in G.nodes(data=True):
        cx = data.get("bbox_cx", None)
        cy = data.get("bbox_cy", None)
        fid = data.get("frame_id", 0)
        if cx is not None and cy is not None:
            # offset each frame slightly so they don't overlap
            pos[node] = (cx + fid * 1.2, 1.0 - cy)
        else:
            pos = nx.spring_layout(G, seed=42, k=2)
            break

    # --- node colors from priority ---
    normalized = [G.nodes[n].get("priority", 0.0) for n in G.nodes()]
    cmap = cm.get_cmap("jet")
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    node_colors = [cmap(norm(v)) for v in normalized]

    # --- node labels ---
    node_labels = {
        node: f"{data.get('label', node)}\nf{data.get('frame_id', '?')}"
        for node, data in G.nodes(data=True)
    }

    # --- split edges by type for different styling ---
    spatial_edges = [(u, v) for u, v, d in G.edges(data=True)
                     if d.get("edge_type") == "spatial"]
    temporal_edges = [(u, v) for u, v, d in G.edges(data=True)
                      if d.get("edge_type") == "temporal"]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=600, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels,
                            font_size=6, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=spatial_edges,
                           edge_color="gray", arrows=True,
                           arrowsize=10, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=temporal_edges,
                           edge_color="steelblue", style="dashed",
                           arrows=True, arrowsize=10, ax=ax)

    # --- edge labels for spatial edges only (temporal would clutter) ---
    spatial_edge_labels = {
        (u, v): f"{d.get('predicate', '')}\n(t{d.get('tier', '?')} c{d.get('confidence', 0):.2f})"
        for u, v, d in G.edges(data=True)
        if d.get("edge_type") == "spatial"
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=spatial_edge_labels,
                                font_size=5, ax=ax,
                                bbox=dict(boxstyle="round,pad=0.2",
                                        fc="white", alpha=0.5))

    temporal_edge_labels = {
        (u, v): d.get("predicate", "same_as")
        for u, v, d in G.edges(data=True)
        if d.get("edge_type") == "temporal"
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=temporal_edge_labels,
                                font_size=5, ax=ax,
                                font_color="steelblue",
                                bbox=dict(boxstyle="round,pad=0.2",
                                        fc="white", alpha=0.5))

    # --- colorbar ---
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=fig.axes[0], label="priority", shrink=0.6)

    # --- legend for edge types ---
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color="gray", label="spatial"),
        Line2D([0], [0], color="steelblue", linestyle="dashed", label="temporal"),
    ]
    ax.legend(handles=legend, loc="upper left")

    ax.set_title(f"Global Knowledge Graph — {G.number_of_nodes()} nodes, "
                 f"{G.number_of_edges()} edges across {len(set(nx.get_node_attributes(G, 'frame_id').values()))} frames")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")


# ---------------------------------------------------------------------------
# 4. Print summary
# ---------------------------------------------------------------------------

def print_summary(G: nx.DiGraph):
    print(f"\nGlobal graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    spatial = sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == "spatial")
    temporal = sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == "temporal")
    print(f"  Spatial edges:  {spatial}")
    print(f"  Temporal edges: {temporal}")

    from collections import Counter
    classes = Counter(d.get("semantic_class", "unknown")
                      for _, d in G.nodes(data=True))
    print(f"  Node classes: {dict(classes)}")

    # most connected nodes
    top = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Most connected nodes:")
    for node, deg in top:
        label = G.nodes[node].get("label", node)
        print(f"    {label} (frame {G.nodes[node].get('frame_id', '?')}): degree {deg}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    frames = load_frame_graphs(JSON_DIR)
    G = build_global_graph(frames)
    print_summary(G)
    render_global_graph(G, OUTPUT_DIR / "global_kg.png")