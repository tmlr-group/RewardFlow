# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import torch
import numpy as np
import random
from typing import List, Tuple, Dict
import math
from PIL import Image
from verl import DataProto
from difflib import SequenceMatcher

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

def to_list_of_dict(batch: DataProto) -> list[dict]:
    tensors = batch.batch
    non_tensor = batch.non_tensor_batch
    batch_size = len(tensors['input_ids'])
    save_list = []
    for bs in range(batch_size):
        save_dict = dict()
        for key, val in tensors.items():
            save_dict[key] = val[bs]
        for key, val in non_tensor.items():
            save_dict[key] = val[bs]
        save_list.append(save_dict)
    return save_list


def torch_to_numpy(tensor, is_object=False):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError(f"Unsupported type: {type(tensor)})")

    if is_object:
        tensor = tensor.astype(object)
    return tensor

def numpy_to_torch(array, device):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor):
        array = array.to(device)
    else:
        raise ValueError(f"Unsupported type: {type(array)})")
    return array


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 256 * 256):
    if isinstance(image, torch.Tensor):
        image = torch_to_numpy(image)
    if image.max() < 1:
        image = image * 255.0
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = Image.fromarray(image)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def adjust_batch(config, data: DataProto, mode="copy") -> DataProto:
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    size_divisor_ref = config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu * world_size
    size_divisor_rollout = config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * world_size
    if "multi_modal_inputs" in data.non_tensor_batch:
        size_divisor_actor = config.actor_rollout_ref.actor.ppo_mini_batch_size
    else:
        size_divisor_actor = config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * world_size
    size_divisor = np.lcm.reduce(np.array([size_divisor_ref, size_divisor_rollout, size_divisor_actor])).item()

    # check if the batch size is divisible by the dp size, if not, delete the last few samples to make it divisible
    bs = len(data)
    remainder = bs % size_divisor
    if remainder == 0:
        return data
    
    if mode == "delete":
        # Generate indices to remove, rather than indices to keep
        remove_indices = np.random.choice(bs, remainder, replace=False)
        # Sort remove_indices to maintain stability when deleting
        remove_indices = np.sort(remove_indices)
        
        # Create a boolean mask for elements to keep
        keep_mask = np.ones(bs, dtype=bool)
        keep_mask[remove_indices] = False

        keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
        # Apply the mask to keep elements in their original order
        tensor_data = data.batch[keep_mask_tensor]
        non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
        del data
    elif mode == "copy":
        to_add = size_divisor - remainder
        dup_indices = np.random.choice(bs, to_add, replace=False)
        dup_proto = data.select_idxs(dup_indices)

        adjusted_batch = DataProto.concat([data, dup_proto])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return adjusted_batch


def filter_group_data(batch_list : List[Dict],
                        episode_rewards: np.ndarray,
                        episode_lengths: np.ndarray,
                        success: Dict[str, np.ndarray],
                        traj_uid: np.ndarray,
                        tool_callings: np.ndarray,
                        config,
                        last_try: bool = False,
                        ):
    """
    Dynamic Sampling:
    Over-sample and filter out episode group in which all episodes have the same rewards.
    Adopted from DAPO (https://arxiv.org/abs/2503.14476)
    """
    if last_try:
        return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    batch_size = config.data.train_batch_size
    group_n = config.env.rollout.n
    if group_n <= 1:
        print("Warning: group_n <= 1, no need to adopt dynamic sampling")

    # Handle each group
    keep_indices = np.array([], dtype=np.int64)
    for i in range(batch_size):
        # Get the indices of the current group
        group_indices = np.arange(i * group_n, (i + 1) * group_n)
        group_rewards = episode_rewards[group_indices]

        # check if all group_traj_uid are the same
        for index in group_indices:
            assert batch_list[index][0]['uid'] == batch_list[group_indices[0]][0]['uid']

        # Check if all rewards in the group are the same
        if not np.all(group_rewards == group_rewards[0]):
            # If so, keep the entire group, otherwise, remove it
            keep_indices = np.concatenate((keep_indices, group_indices))
    
    # Filter the batch_list, episode_rewards, episode_lengths, success, and tool_callings based on the keep_indices
    success = {
        key: value[keep_indices]
        for key, value in success.items()
        if len(value) == len(batch_list)
    }
    batch_list = [batch_list[i] for i in keep_indices]
    episode_rewards = episode_rewards[keep_indices]
    episode_lengths = episode_lengths[keep_indices]
    # success = {key: value[keep_indices] for key, value in success.items()}
    traj_uid = traj_uid[keep_indices]
    tool_callings = tool_callings[keep_indices]

    return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings


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

def extract_unique_states(state_list, use_similarity=False, similarity_threshold=0.95):

    if not use_similarity:
        unique_state_list = []
        # Use a set for efficient checking of unique states
        unique_state_hashes = set()

        for traj in state_list:
            for step in traj:
                state_arr = step['state']
                state_hash = to_hashable(state_arr)
                if state_hash not in unique_state_hashes:
                    unique_state_list.append(state_arr)
                    unique_state_hashes.add(state_hash)
        
        # Map unique states to indices
        state_to_idx = {to_hashable(state): i for i, state in enumerate(unique_state_list)}
        idx_to_state = {i: state for i, state in enumerate(unique_state_list)}
        return unique_state_list, state_to_idx, idx_to_state

    else: 
        clusters = []
        
        all_states = []
        for traj in state_list:
            for step in traj:
                all_states.append(step['state'])

        for state_arr in all_states:
            if not clusters:
                clusters.append([state_arr])
                continue

            found_cluster = False
            for cluster in clusters:
                # Check similarity with the representative of the cluster (the first element)
                if are_similar(state_arr, cluster[0], similarity_threshold):
                    cluster.append(state_arr)
                    found_cluster = True
                    break
            
            # If not found cluster, create a new one
            if not found_cluster:
                clusters.append([state_arr])
        
        
        unique_state_list = clusters
        
        # Map states to cluster indices
        state_to_idx = {}
        for i, cluster in enumerate(clusters):
            for state in cluster:
                state_hash = to_hashable(state)
                state_to_idx[state_hash] = i

        # Reverse mapping from index to cluster
        idx_to_state = {i: cluster for i, cluster in enumerate(clusters)}
        
        return unique_state_list, state_to_idx, idx_to_state


def build_trajectory(state_list, action_list, state_to_idx):
    """
    Generate trajectory (list of triplets): (src_idx, action_label, dst_idx, src_reward, dst_reward)
    """
    trajectory = []
    for i in range(len(state_list)):
        trajectory.append([])
        for j in range(len(state_list[i]) - 1):
            src_state = state_list[i][j]['state']
            dst_state = state_list[i][j + 1]['state']
            src_idx = state_to_idx[to_hashable(src_state)]
            dst_idx = state_to_idx[to_hashable(dst_state)]
            action_label = action_list[i][j]
            src_reward = state_list[i][j]['reward']
            dst_reward = state_list[i][j + 1]['reward']
            trajectory[i].append((src_idx, action_label, dst_idx, src_reward, dst_reward))
    return trajectory

def unique_trajectory(trajectory):
    unique_trajectory = []
    for i in range(len(trajectory)):
        unique_trajectory.append([])
        for item in trajectory[i]:
            if item not in unique_trajectory[i]:
                unique_trajectory[i].append(item)
    flat_unique_trajectory = [item for sublist in unique_trajectory for item in sublist]
    return flat_unique_trajectory

def clean_trajectory(trajectory, idx_to_state):
    cleaned_trajectory = []
    for i in range(len(trajectory)):
        cleaned_trajectory.append([])
        for triple in trajectory[i]:
            if triple[0] != triple[2]:
                cleaned_trajectory[i].append(triple)
    return cleaned_trajectory

def get_personalization(flat_unique_trajectory):
    reward_dict = {}
    for (src, action, dst, src_reward, dst_reward) in flat_unique_trajectory:
        if src not in reward_dict:
            reward_dict[src] = src_reward
        if dst not in reward_dict:
            reward_dict[dst] = dst_reward

    return reward_dict
