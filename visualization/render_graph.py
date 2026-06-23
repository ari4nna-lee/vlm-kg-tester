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

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return img