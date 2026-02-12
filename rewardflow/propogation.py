import networkx as nx
from collections import deque

def propagate_reward_decay(flat_unique_trajectory, gamma=0.95, max_iter=500):
    G = nx.DiGraph()
    for (src, action, dst, src_reward, dst_reward) in flat_unique_trajectory:
        G.add_edge(src, dst, action=action, label=action)

    reward_dict = {}
    for (src, action, dst, src_reward, dst_reward) in flat_unique_trajectory:
        if src not in reward_dict:
            reward_dict[src] = src_reward
        if dst not in reward_dict:
            reward_dict[dst] = dst_reward

    value_dict = {node: reward for node, reward in reward_dict.items()}
    for _ in range(max_iter):
        new_value_dict = value_dict.copy()
        for node in G.nodes:
            max_value = value_dict[node]
            for _, succ in G.out_edges(node):
                propagated = gamma * value_dict[succ]
                if propagated > max_value:
                    max_value = propagated
            new_value_dict[node] = max_value
        value_dict = new_value_dict
    return G, value_dict
