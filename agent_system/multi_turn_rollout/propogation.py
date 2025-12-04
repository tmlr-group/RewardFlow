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


def propagate_reward_pagerank(flat_unique_trajectory, alpha=0.7, personalization=None, max_iter=500):
    """
    使用 PageRank 算法传播 reward，返回图 G 和每个节点的 pagerank value。

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
    # PageRank 传播
    pagerank_dict = nx.pagerank(G, alpha=alpha, personalization=personalization, max_iter=max_iter)

    return G, pagerank_dict

import math

def _build_graph_and_rewards(flat_unique_trajectory):
    G = nx.DiGraph()
    for (src, action, dst, src_reward, dst_reward) in flat_unique_trajectory:
        G.add_edge(src, dst, action=action, label=action)
    reward_dict = {}
    for (src, action, dst, src_reward, dst_reward) in flat_unique_trajectory:
        if src not in reward_dict:
            reward_dict[src] = src_reward
        if dst not in reward_dict:
            reward_dict[dst] = dst_reward
    return G, reward_dict

def _normalize_rewards(reward_dict):
    min_r = min(reward_dict.values())
    max_r = max(reward_dict.values())
    if max_r > min_r:
        return {n: (r - min_r) / (max_r - min_r) for n, r in reward_dict.items()}
    # 退化情形：全相等
    return {n: 1.0 for n in reward_dict}

def propagate_reward_random_walk_stationary(flat_unique_trajectory, max_iter=100, tol=1e-8, dangling_weight='uniform'):
    """
    计算普通随机游走的稳态分布 pi（无重启）。返回 (G, pi_dict)。
    - dangling_weight:
        - 'uniform': 悬挂节点的概率均匀分配到所有节点
        - 'self': 悬挂节点留在自身
    """
    G, reward_dict = _build_graph_and_rewards(flat_unique_trajectory)
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return G, {}

    # 初始分布均匀
    pi = {u: 1.0 / n for u in nodes}

    # 预处理转移概率：对每个节点，按出边均分
    outdeg = {u: G.out_degree(u) for u in nodes}

    for _ in range(max_iter):
        new_pi = {u: 0.0 for u in nodes}
        for u in nodes:
            if outdeg[u] > 0:
                p = 1.0 / outdeg[u]
                for _, v in G.out_edges(u):
                    new_pi[v] += pi[u] * p
            else:
                # 悬挂节点处理
                if dangling_weight == 'uniform':
                    w = 1.0 / n
                    for v in nodes:
                        new_pi[v] += pi[u] * w
                else:  # 'self'
                    new_pi[u] += pi[u]

        # 归一化（数值稳健）
        s = sum(new_pi.values())
        if s > 0:
            for u in nodes:
                new_pi[u] /= s

        # 收敛判定
        delta = sum(abs(new_pi[u] - pi[u]) for u in nodes)
        pi = new_pi
        if delta < tol:
            break

    return G, pi

def propagate_reward_bfs(flat_unique_trajectory, gamma=0.9, direction='backward', max_depth=None, aggregate='max'):
    """
    基于 BFS 的图传播（按层传播，距离 d 的贡献为 reward * gamma**d）。
    返回 (G, value_dict)。

    - direction: 'backward'（默认，沿反向边传播，适合从终点回传）或 'forward'（沿正向边向前传）。
    - max_depth: 限制传播的最大层数（距离），None 表示不限制。
    - aggregate: 'max' 或 'sum'
        - 'max': 对每个节点取所有源贡献的最大值（基于距离衰减和值大小）
        - 'sum': 累加所有源对该节点的贡献

    flat_unique_trajectory: List[Tuple[src, action, dst, src_reward, dst_reward]]
    """
    # 构图 & 奖励表
    G = nx.DiGraph()
    for (src, action, dst, src_reward, dst_reward) in flat_unique_trajectory:
        G.add_edge(src, dst, action=action, label=action)

    reward_dict = {}
    for (src, action, dst, src_reward, dst_reward) in flat_unique_trajectory:
        if src not in reward_dict:
            reward_dict[src] = src_reward
        if dst not in reward_dict:
            reward_dict[dst] = dst_reward

    nodes = list(G.nodes())
    if not nodes:
        return G, {}

    # 选择传播方向的邻接访问器
    if direction == 'backward':
        # 反向传播：使用前驱（入边）邻接
        neighbors = lambda u: (v for v, _ in G.in_edges(u))
    elif direction == 'forward':
        # 正向传播：使用后继（出边）邻接
        neighbors = lambda u: (v for _, v in G.out_edges(u))
    else:
        raise ValueError("direction must be 'backward' or 'forward'")

    # 选择种子节点：默认 reward > 0 的节点；如无，则全部节点
    seed_nodes = [n for n, r in reward_dict.items() if r > 0]
    if not seed_nodes:
        seed_nodes = nodes

    # 初始化输出
    if aggregate == 'sum':
        value_dict = {n: 0.0 for n in nodes}
    elif aggregate == 'max':
        value_dict = {n: float('-inf') for n in nodes}
    else:
        raise ValueError("aggregate must be 'max' or 'sum'")

    # 对每个种子做 BFS，计算对所有节点的贡献
    for s in seed_nodes:
        base_reward = reward_dict.get(s, 0.0)
        if base_reward == 0 and len(seed_nodes) > 1:
            # 无贡献的种子可跳过（若只有这一个种子，仍然允许它传播 0）
            continue

        # BFS 队列与最短距离
        dist = {s: 0}
        q = deque([s])

        # 对种子自身的贡献（距离 0）
        contrib0 = base_reward  # gamma ** 0 == 1
        if aggregate == 'sum':
            value_dict[s] += contrib0
        else:  # 'max'
            if contrib0 > value_dict[s]:
                value_dict[s] = contrib0

        while q:
            u = q.popleft()
            d = dist[u]
            if max_depth is not None and d >= max_depth:
                continue
            nd = d + 1
            for v in neighbors(u):
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    q.append(v)

        # 汇总该种子对所有可达节点的贡献（按最短距离衰减）
        for v, d in dist.items():
            if d == 0:
                continue  # 已处理
            contrib = base_reward * (gamma ** d)
            if aggregate == 'sum':
                value_dict[v] += contrib
            else:
                if contrib > value_dict[v]:
                    value_dict[v] = contrib

    # 将未被任何种子触达的节点，从 -inf 修正为其自身 reward（或 0）
    if aggregate == 'max':
        for n in nodes:
            if value_dict[n] == float('-inf'):
                value_dict[n] = reward_dict.get(n, 0.0)

    return G, value_dict


def propagate_reward_random_sampling(
    flat_unique_trajectory,
    gamma=0.9,
    num_samples=100,
    max_steps=20,
    direction='backward',
    aggregate='sum',
    seed_strategy='positive',
    with_restart=False,
    restart=0.15,
):
    """
    随机采样的图传播（Monte Carlo Random Sampling）。
    从种子节点出发做多次随机游走，把每一步对访问节点的贡献累积。
    返回 (G, value_dict)。

    - gamma: 每前进一步的折扣因子（贡献乘以 gamma^t）
    - num_samples: 每个种子的采样条数
    - max_steps: 单条随机游走的最大步数
    - direction: 'backward'（沿入边回传）或 'forward'（沿出边扩散）
    - aggregate: 'sum'（累加所有贡献）或 'max'（取最大贡献）
    - seed_strategy: 'positive'（reward>0 的节点为种子）或 'all'（所有节点为种子）
    - with_restart: 是否带重启；若为 True，每步以概率 restart 回到当前种子
    - restart: 重启概率（with_restart=True 时生效）
    """
    import random
    G = nx.DiGraph()
    for (src, action, dst, src_reward, dst_reward) in flat_unique_trajectory:
        G.add_edge(src, dst, action=action, label=action)

    # 收集 reward
    reward_dict = {}
    for (src, action, dst, src_reward, dst_reward) in flat_unique_trajectory:
        if src not in reward_dict:
            reward_dict[src] = src_reward
        if dst not in reward_dict:
            reward_dict[dst] = dst_reward

    nodes = list(G.nodes())
    if not nodes:
        return G, {}

    # 传播方向
    if direction == 'backward':
        neigh = lambda u: [v for v, _ in G.in_edges(u)]
    elif direction == 'forward':
        neigh = lambda u: [v for _, v in G.out_edges(u)]
    else:
        raise ValueError("direction must be 'backward' or 'forward'")

    # 种子
    if seed_strategy == 'positive':
        seeds = [n for n, r in reward_dict.items() if r > 0]
        if not seeds:
            seeds = nodes
    elif seed_strategy == 'all':
        seeds = nodes
    else:
        raise ValueError("seed_strategy must be 'positive' or 'all'")

    # 聚合器
    if aggregate == 'sum':
        value_dict = {n: 0.0 for n in nodes}
        def apply(v, x): return v + x
        def init_val(): return 0.0
    elif aggregate == 'max':
        value_dict = {n: float('-inf') for n in nodes}
        def apply(v, x): return x if x > v else v
        def init_val(): return float('-inf')
    else:
        raise ValueError("aggregate must be 'sum' or 'max'")

    # 逐种子采样
    for s in seeds:
        base_reward = reward_dict.get(s, 0.0)
        # 若多种子且该种子 reward 为 0，可选择跳过以加速
        if base_reward == 0.0 and len(seeds) > 1:
            continue

        for _ in range(num_samples):
            cur = s
            # 步 0：对种子自身贡献
            contrib = base_reward  # gamma^0
            value_dict[cur] = apply(value_dict[cur], contrib)

            for t in range(1, max_steps + 1):
                # 重启
                if with_restart and random.random() < restart:
                    cur = s

                nbrs = neigh(cur)
                if not nbrs:
                    break  # 悬挂节点，终止该条游走

                cur = random.choice(nbrs)
                contrib = base_reward * (gamma ** t)
                value_dict[cur] = apply(value_dict[cur], contrib)

    # 修正 max 聚合中未触达节点
    if aggregate == 'max':
        for n in nodes:
            if value_dict[n] == float('-inf'):
                value_dict[n] = reward_dict.get(n, 0.0)

    return G, value_dict