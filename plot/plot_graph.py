import networkx as nx
import matplotlib.pyplot as plt

import os
import numpy as np
from PIL import Image

def show_state(rgb_array: np.ndarray):
    """
    显示单帧 RGB 状态。rgb_array 是 numpy 数组。
    """
    rgb_array = np.array(rgb_array, dtype=np.uint8)
    img = Image.fromarray(rgb_array)
    img.show()

def plot_transition_graph(flat_unique_trajectory, title_suffix="(Before Propagation)", save_path="tmp/transition_graph.png", save=True):
    """
    绘制状态转移图，并将图片保存到指定路径（默认 tmp/transition_graph.png）。
    """
    G = nx.DiGraph()
    node_rewards = {}
    for (src, action, dst, src_reward, dst_reward) in flat_unique_trajectory:
        G.add_edge(src, dst, label=action)
        if src not in node_rewards:
            node_rewards[src] = src_reward
        if dst not in node_rewards:
            node_rewards[dst] = dst_reward

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800, arrowsize=20,
            connectionstyle='arc3,rad=0.1')
    for node, (x, y) in pos.items():
        reward = node_rewards.get(node, None)
        if reward is not None:
            plt.text(x, y + 0.08, f"R={reward}", fontsize=10, ha='center', va='bottom', color='green')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title(f"Sokoban State Transition Graph {title_suffix}")

    # 保存图片
    if save:
        os.makedirs("tmp", exist_ok=True)
        save_path = os.path.join("tmp", f"transition_graph_{title_suffix}.png")
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    return G, pos


def plot_propagated_values(G, pos, value_dict, title_suffix="(After Propagation)", save_path="tmp/propagated_values.png", save=True):
    """
    绘制传播后的值图，并将图片保存到指定路径（默认 tmp/propagated_values.png）。
    """
    node_to_value = {node: value_dict.get(node, 0.0) for node in G.nodes}
    for node in G.nodes:
        G.nodes[node]['propagated_value'] = node_to_value[node]

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800, arrowsize=20,
            connectionstyle='arc3,rad=0.1')
    for node, (x, y) in pos.items():
        propagated = node_to_value.get(node, None)
        if propagated is not None:
            plt.text(x, y + 0.08, f"V={propagated:.2f}", fontsize=10, ha='center', va='bottom', color='blue')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title(f"Sokoban State Transition Graph {title_suffix}")

    # 保存图片
    if save:
        os.makedirs("tmp", exist_ok=True)
        save_path = os.path.join("tmp", f"propagated_values_{title_suffix}.png")
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()