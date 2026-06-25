import matplotlib
matplotlib.use("Agg")  # must be before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np

def render_graph(G, size=(6, 6)):
    fig = plt.figure(figsize=size)

    # --- node placement ---
    pos = {}
    for node, data in G.nodes(data=True):
        cx = data.get("bbox_cx", None)
        cy = data.get("bbox_cy", None)
        if cx is not None and cy is not None:
            pos[node] = (cx, 1.0 - cy)
        else:
            pos = nx.spring_layout(G, seed=42)
            break

    # --- node colors from priority via JET colormap ---
    priorities = [
        G.nodes[node].get("priority", 0.0)
        for node in G.nodes()
    ]
    if priorities:
        p_min, p_max = min(priorities), max(priorities)
        p_range = p_max - p_min if p_max > p_min else 1.0
        normalized = [(p - p_min) / p_range for p in priorities]
    else:
        normalized = []

    cmap = cm.get_cmap("jet")
    node_colors = [cmap(v) for v in normalized]

    # --- labels ---
    node_labels = {
        node: data.get("label", data.get("semantic_class", node))
        for node, data in G.nodes(data=True)
    }
    edge_labels = {
        (u, v): data.get("relation", data.get("predicate", ""))
        for u, v, data in G.edges(data=True)
    }

    nx.draw(
        G,
        pos,
        labels=node_labels,
        with_labels=True,
        node_size=700,
        font_size=8,
        node_color=node_colors,
        edge_color="gray",
        arrows=True,
    )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=6,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6),
    )

    # --- colorbar legend ---
    sm = cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=p_min if priorities else 0,
                           vmax=p_max if priorities else 1)
    )
    sm.set_array([])
    plt.colorbar(sm, ax=fig.axes[0], label="priority", shrink=0.6)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    img = img[:, :, :3]
    plt.close(fig)
    return img