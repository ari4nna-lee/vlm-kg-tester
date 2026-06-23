import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def render_graph(G, size=(6, 6)):
    fig = plt.figure(figsize=size)
    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=700,
        font_size=8
    )

    fig.canvas.draw()

    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    fig.canvas.draw()

    img = np.asarray(fig.canvas.buffer_rgba())
    img = img[:, :, :3]  # drop alpha

    plt.close(fig)
    return img