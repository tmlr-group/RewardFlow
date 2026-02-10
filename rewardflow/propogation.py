import networkx as nx
from collections import deque

def propagate_reward_decay(flat_unique_trajectory, gamma=0.95, max_iter=500):
    """
    对 flat_unique_trajectory 中的奖励进行 gamma 衰减传播，返回图 G 和每个节点的 propagated value。
    对 reward 做归一化处理。
    flat_unique_trajectory: List[Tuple[src, action, dst, src_reward, dst_reward]]
    """
    G = nx.DiGraph()
    for (src, action, dst, src_reward, dst_reward) in flat_unique_trajectory:
        # 添加双向有向边：src <-> dst
        G.add_edge(src, dst, action=action, label=action)
        G.add_edge(dst, src, action=action, label=action)

    # 收集每个节点的原始 reward（出现的第一次）
    reward_dict = {}
    for (src, action, dst, src_reward, dst_reward) in flat_unique_trajectory:
        if src not in reward_dict:
            reward_dict[src] = src_reward
        if dst not in reward_dict:
            reward_dict[dst] = dst_reward

    # 值迭代风格的最大传播
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
