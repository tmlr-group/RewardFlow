# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (RewardFlow) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Core functions to implement RewardFlow algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to implement RewardFlow
"""

import numpy as np
import torch
from collections import defaultdict, Counter
from verl import DataProto
import uuid
import networkx as nx
from collections import deque
from difflib import SequenceMatcher
from typing import Sequence, List, Dict, Any
import re

# ---------------------------------------------------------- #
# ------------ General Functions of RewardFlow ------------- #
# ---------------------------------------------------------- #
def to_hashable(x):
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
            


def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Check whether two text observations are similar enough.
    
    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.
    
    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO in this version.")
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_returns(batch: DataProto):
    # 直接返回原始rewards，保持数据结构正确
    rewards = batch.non_tensor_batch['step_rewards'].astype(np.float32)
    all_returns = torch.tensor(rewards, dtype=torch.float32, device=batch.batch['input_ids'].device)
    
    return all_returns

# ---------------------------------------------------------- #
# -------------- Core Functions of RewardFlow -------------- #
# ---------------------------------------------------------- #

def compute_rewardflow_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   ):
    
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode-level group reward
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)
    
    # Compute step_group_uids
    step_group_uids = build_step_group(anchor_obs, index)

    # Compute step-level group reward
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    scores = episode_advantages + step_advantage_w * step_advantages
    # scores = step_advantages
    print("solely use step-level advantage")
    return scores, scores


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_all_data: bool = True,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for RewardFlow.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        compute_mean_std_cross_all_data: bool
            If True (more stable), the mean and std are computed across all data in the batch. 
            If False (i.e., standard episode-level adv), the mean and std are computed across N trajectories.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_all_data:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of episode_group_uid
    summarize : bool
        Whether to summarize the group sizes (default: True)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size = []
    # Process each unique index
    for idx in unique_indices:
        # Get all observations for this index using np.where
        indices = np.where(index == idx)[0]
        obs_group = anchor_obs[indices]
        
        # Create clusters for identical observations
        clusters = defaultdict(list)
        for i, obs in enumerate(obs_group):
            clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
        
        # Assign unique step_group_uid to each cluster
        for obs, original_indices in clusters.items():
            # Generate a UUID for this cluster
            uid = str(uuid.uuid4())
            
            # Assign the same step_group_uid to all elements in this cluster
            group_size.append(len(original_indices))
            for original_idx in original_indices:
                step_group_uids[original_idx] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for RewardFlow.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    # import pickle
    # with open("step_norm_scores.pkl", "wb") as f:
    #     pickle.dump(scores.cpu().numpy(), f)

    # raise Exception("rewardflow test")

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages

def general_clean_trajectory(trajectory, idx_to_state):
    print("general_clean_trajectory: no denoise")
    return trajectory

# ---------------------------------------------------------- #
# ------------ Reward Propagation Functions --------------- #
# ---------------------------------------------------------- #




def apply_rewardflow_propagation(total_batch_list, 
                                 config,
                                 tokenizer,
                                 envs):
    """
    Main function to compute RewardFlow propagation for multi-turn trajectories.
    
    Args:
        total_batch_list: List of trajectory batches
        config: Configuration object with algorithm settings
        tokenizer: Tokenizer for decoding responses
        envs: Environment object with projection function
        gamma: Discount factor
        
    Returns:
        total_batch_list: Updated batch list with propagated rewards
    """
    # Import functions from their respective modules
    from agent_system.multi_turn_rollout.utils import (
        extract_unique_states, 
        build_trajectory, 
        get_personalization,
        unique_trajectory
    )
    from agent_system.multi_turn_rollout.propogation import (
        propagate_reward_decay,
        propagate_reward_pagerank
    )


    raw_state_list, raw_action_list = envs.state_preprocess(total_batch_list, tokenizer)
    
    # Group trajectories by UID
    uid_list = []
    for i in range(len(total_batch_list)):
        for j in range(len(total_batch_list[0])):
            uid_list.append(total_batch_list[i][j]['uid'])
            break
    
    # Find first and last indices for each group of identical elements
    first_last_indices = []
    if uid_list:
        prev = uid_list[0]
        start = 0
        for idx in range(1, len(uid_list)):
            if uid_list[idx] != prev:
                first_last_indices.append((start, idx))
                start = idx
                prev = uid_list[idx]
        first_last_indices.append((start, len(uid_list)))

    # Process each group for reward propagation
    value_dict_list = []
    idx_to_state_list = []
    state_to_idx_list = []
    flat_unique_trajectory_list = []
    # raise ValueError("stop here")
    for (start_idx, end_idx) in first_last_indices:
        selected_state_list = raw_state_list[start_idx:end_idx]
        selected_action_list = raw_action_list[start_idx:end_idx]
        _, state_to_idx, idx_to_state = extract_unique_states(selected_state_list)
        trajectory = build_trajectory(selected_state_list, selected_action_list, state_to_idx)
        # cleaned_trajectory = envs.clean_trajectory(trajectory, idx_to_state)
        cleaned_trajectory = envs.clean_trajectory(trajectory, idx_to_state)
        # ablation: no valid action clean
        # cleaned_trajectory = general_clean_trajectory(trajectory, idx_to_state)
        flat_unique_trajectory = unique_trajectory(cleaned_trajectory)
        flat_unique_trajectory_list.append(flat_unique_trajectory)
        # Apply reward propagation based on config
        if config.algorithm.rewardflow.propagate == "decay":
            G_propagate, value_dict = propagate_reward_decay(
                flat_unique_trajectory, 
                gamma=config.algorithm.gamma, 
                max_iter=1000
            )
        elif config.algorithm.rewardflow.propagate in ["pr", "ppr"]:
            if config.algorithm.rewardflow.propagate == "pr":
                personalization = None
            elif config.algorithm.rewardflow.propagate == "ppr":
                personalization = get_personalization(flat_unique_trajectory)
                if all(value == 0 for value in personalization.values()):
                    personalization = None
            G_propagate, value_dict = propagate_reward_pagerank(
                flat_unique_trajectory, 
                alpha=config.algorithm.rewardflow.alpha, 
                personalization=personalization, 
                max_iter=1000
            )
            
        else:
            raise ValueError(f"Unknown propagate method: {config.algorithm.rewardflow.propagate}")
        
        for idx in range(len(idx_to_state)):
            if idx not in value_dict:
                value_dict[idx] = 0.0

        value_dict_list.append(value_dict)
        idx_to_state_list.append(idx_to_state)
        state_to_idx_list.append(state_to_idx)
    # Update rewards in trajectories
    new_reward_list = []
    new_reward_abs_list = []
    for i in range(len(total_batch_list)):
        batch_idx = i * len(state_to_idx_list) // len(total_batch_list)
        initial_state = to_hashable(total_batch_list[i][0]['anchor_obs'])
        new_reward = []
        new_reward_abs = []
        
        try:
            state_idx = state_to_idx_list[batch_idx][initial_state]
        except KeyError as e:
            print(f"KeyError: initial_state {initial_state} not found in state_to_idx_list[{batch_idx}]")
            print(f"Available keys: {list(state_to_idx_list[batch_idx].keys())}")
            raise e
        try:
            prev_reward = value_dict_list[batch_idx][state_idx]
        except KeyError as e:
            print(f"KeyError: state_idx {state_idx} not found in value_dict_list[{batch_idx}]")
            print(f"Available keys: {list(value_dict_list[batch_idx].keys())}")
            raise e
        
        for j in range(len(total_batch_list[i])-1):
            if not total_batch_list[i][j]['active_masks']:
                break
            # Assign to_state reward to the action
            cur_state = to_hashable(total_batch_list[i][j+1]['anchor_obs'])
            state_idx = state_to_idx_list[batch_idx][cur_state]
            cur_reward = value_dict_list[batch_idx][state_idx]
            

            # # smooth the reward
            # if cur_reward == 0.0:
            #     cur_reward = prev_reward

            new_reward.append(cur_reward - prev_reward)

            new_reward_abs.append(cur_reward)
            prev_reward = cur_reward
        new_reward_abs_list.append(new_reward_abs)
        new_reward_list.append(new_reward)

    # Update the original batch with new rewards
    for i in range(len(total_batch_list)):
        for j in range(len(total_batch_list[i])):
            # 为防止报错，把后续失效的reward补齐
            if j < len(new_reward_list[i]):
                total_batch_list[i][j]['step_rewards'] = new_reward_list[i][j]
                total_batch_list[i][j]['step_rewards_abs'] = new_reward_abs_list[i][j]
            else:
                total_batch_list[i][j]['step_rewards'] = total_batch_list[i][j]['rewards'] 
                total_batch_list[i][j]['step_rewards_abs'] = total_batch_list[i][j]['rewards'] 
    print("length of total_batch_list: ", len(total_batch_list))
    # return total_batch_list, flat_unique_trajectory_list, value_dict_list, idx_to_state_list, state_to_idx_list
    return total_batch_list

