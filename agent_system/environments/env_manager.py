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

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
import re
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask

    def state_preprocess(self, total_batch_list, tokenizer):

        raw_state_list = []
        raw_action_list = []

        for traj in total_batch_list:
            # Initialize state and action sequences for each trajectory
            state_seq = [{
                "state": traj[0]['anchor_obs'],
                "reward": 0,
                "active_masks": traj[0]['active_masks']
            }]
            action_seq = []

            for j in range(1, len(traj)):
                # If current step's active_masks is False, trajectory ends
                if not traj[j]['active_masks']:
                    # If not the last step (success), add the last valid state and action
                    state_seq.append({
                        "state": traj[j]['anchor_obs'],
                        "reward": traj[j-1]['rewards'],
                        "active_masks": traj[j-1]['active_masks']
                    })
                    raw_response = tokenizer.decode(traj[j-1]['responses'])
                    action, valid = self.projection_f([raw_response])
                    action_seq.append(str(action[0]))
                    break

                # Normal progression, reward comes from previous step
                state_seq.append({
                    "state": traj[j]['anchor_obs'],
                    "reward": traj[j-1]['rewards'],
                    "active_masks": traj[j-1]['active_masks']
                })
                raw_response = tokenizer.decode(traj[j-1]['responses'])
                action, valid = self.projection_f([raw_response])
                action_seq.append(str(action[0]))

            raw_state_list.append(state_seq)
            raw_action_list.append(action_seq)

        return raw_state_list, raw_action_list

    def clean_trajectory(self, trajectory, idx_to_state):
        cleaned_trajectory = []
        for i in range(len(trajectory)):
            cleaned_trajectory.append([])
            for triple in trajectory[i]:
                if triple[0] != triple[2]:
                    cleaned_trajectory[i].append(triple)
        return cleaned_trajectory

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break

    
    def state_preprocess(self, total_batch_list, tokenizer):
        # 将正则表达式编译移到类级别，避免重复编译
        if not hasattr(self, '_compiled_patterns'):
            self._compiled_patterns = {
                'cool': re.compile(r"You cool the (.*?) using"),
                'heat': re.compile(r"You heat the (.*?) using"),
                'clean': re.compile(r"You clean the (.*?) using"),
                'object': re.compile(r"((?:[a-zA-Z]+ ?)+ \d+)"),
                'a_prefix': re.compile(r"\ba ([a-zA-Z]+ ?\d+)")
            }
        
        patterns = self._compiled_patterns
        success_state = "You Won!"
        
        # 预编译字符串模板
        state_template = "{} {}"
        object_state_template = "{} ({}ed)"
        
        # 批量解码所有responses，避免重复调用tokenizer.decode
        all_responses = []
        response_positions = []
        for i, traj in enumerate(total_batch_list):
            for j in range(1, len(traj)):
                if not traj[j]['active_masks']:
                    all_responses.append(traj[j-1]['responses'])
                    response_positions.append((i, j-1, True))  # (traj_idx, step_idx, is_final)
                    break
                all_responses.append(traj[j-1]['responses'])
                response_positions.append((i, j-1, False))
        
        # 批量解码
        if all_responses:
            decoded_responses = tokenizer.batch_decode(all_responses)
            # 批量处理projection
            actions, valids = self.projection_f(decoded_responses, self.envs.get_admissible_commands)
        else:
            decoded_responses = []
            actions = []
            valids = []
        
        # 创建响应映射
        response_map = {}
        for idx, (i, j, is_final) in enumerate(response_positions):
            response_map[(i, j, is_final)] = str(actions[idx]) if idx < len(actions) else ""
        
        raw_state_list = []
        raw_action_list = []
        total_state_list = []
        
        response_idx = 0
        
        for i, traj in enumerate(total_batch_list):
            object_state_dict = {
                "cool": set(),
                "heat": set(),
                "clean": set(),
            }
            
            # 预计算初始状态
            initial_state = state_template.format(traj[0]['anchor_obs'], traj[0]['inventory_state'])
            state_seq = [{
                "state": initial_state,
                "reward": 0,
                "active_masks": traj[0]['active_masks']
            }]
            
            total_batch_list[i][0]['anchor_obs'] = initial_state
            action_seq = []
            
            for j in range(1, len(traj)):
                if not traj[j]['active_masks']:
                    state_seq.append({
                        "state": success_state,
                        "reward": traj[j-1]['rewards'],
                        "active_masks": traj[j-1]['active_masks']
                    })
                    
                    total_batch_list[i][j]['anchor_obs'] = success_state
                    action_seq.append(response_map.get((i, j-1, True), ""))
                    break
                
                # 优化对象状态更新：使用单一循环
                anchor_obs = traj[j]['anchor_obs']
                current_obj = None
                current_action = None
                
                # 检查所有模式，找到匹配的
                for action, pattern in [('cool', patterns['cool']), ('heat', patterns['heat']), ('clean', patterns['clean'])]:
                    match = pattern.search(anchor_obs)
                    if match:
                        current_obj = match.group(1)
                        current_action = action
                        break
                
                # 更新对象状态
                if current_obj and current_action:
                    # 从其他状态中移除
                    for other_action in object_state_dict:
                        if other_action != current_action:
                            object_state_dict[other_action].discard(current_obj)
                    object_state_dict[current_action].add(current_obj)
                
                # 处理inventory状态
                inventory_state = traj[j]['inventory_state']
                
                # 优化字符串处理：只在需要时进行替换
                if current_obj and current_action:
                    # 移除"a "前缀
                    inventory_state_no_a = patterns['a_prefix'].sub(r"\1", inventory_state)
                    objects = patterns['object'].findall(inventory_state_no_a)
                    
                    if objects:
                        # 使用字典查找优化状态匹配
                        obj_state_map = {}
                        for action_key, obj_set in object_state_dict.items():
                            for obj in obj_set:
                                obj_state_map[obj] = action_key
                        
                        # 构建替换映射
                        replace_map = {}
                        for obj in objects:
                            if obj in obj_state_map:
                                state = obj_state_map[obj]
                                replace_map[obj] = object_state_template.format(obj, state)
                        
                        # 执行替换
                        if replace_map:
                            def obj_replace(match):
                                return replace_map.get(match.group(0), match.group(0))
                            inventory_state = patterns['object'].sub(obj_replace, inventory_state_no_a)
                
                traj[j]['inventory_state'] = inventory_state
                current_state = state_template.format(traj[j]['anchor_obs'], inventory_state)
                
                state_seq.append({
                    "state": current_state,
                    "reward": traj[j-1]['rewards'],
                    "active_masks": traj[j-1]['active_masks']
                })
                total_batch_list[i][j]['anchor_obs'] = current_state
                action_seq.append(response_map.get((i, j-1, False), ""))
            
            # 将set转为list，保证数据结构一致性
            total_state_list.append({k: list(v) for k, v in object_state_dict.items()})
            raw_state_list.append(state_seq)
            raw_action_list.append(action_seq)
        print("end state_preprocess")
        return raw_state_list, raw_action_list

    def clean_trajectory(self, trajectory, idx_to_state):
        # 1. inventory
        # 2. look
        # 3. examine
        # 4. 无效动作 (下一个状态是Nothing happens)
        cleaned_trajectory = []
        for i in range(len(trajectory)):
            cleaned_trajectory.append([])
            cur_from_state = None
            j = 0
            while j < len(trajectory[i]):
                from_state, action, to_state, from_reward, to_reward = trajectory[i][j]
                if action in ["inventory", "look"] or "examine" in action or "Nothing happens" in idx_to_state[to_state] or from_state == to_state:
                    cur_from_state = from_state
                    # 向下寻找下一个有效的action
                    while j < len(trajectory[i]):
                        from_state, action, to_state, from_reward, to_reward = trajectory[i][j]
                        # 注意这里要判断是当前定位的from_state (cur_from_state)和to_state是否相同
                        if action in ["inventory", "look"] or "examine" in action or "Nothing happens" in idx_to_state[to_state] or cur_from_state == to_state:
                            j += 1
                        else:
                            cleaned_trajectory[i].append((cur_from_state, action, to_state, from_reward, to_reward))
                            j += 1
                            break
                else:
                    cleaned_trajectory[i].append((from_state, action, to_state, from_reward, to_reward))
                    j += 1

        return cleaned_trajectory



class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    
    def state_preprocess(self, total_batch_list, tokenizer):

        raw_state_list = []
        raw_action_list = []

        for traj in total_batch_list:
            # Initialize state and action sequences for each trajectory
            state_seq = [{
                "state": traj[0]['anchor_obs'],
                "reward": 0,
                "active_masks": traj[0]['active_masks']
            }]
            action_seq = []

            for j in range(1, len(traj)):
                # If current step's active_masks is False, trajectory ends
                if not traj[j]['active_masks']:
                    # If not the last step (success), add the last valid state and action
                    state_seq.append({
                        "state": traj[j]['anchor_obs'],
                        "reward": traj[j-1]['rewards'],
                        "active_masks": traj[j-1]['active_masks']
                    })
                    raw_response = tokenizer.decode(traj[j-1]['responses'])
                    action, valid = self.projection_f([raw_response])
                    action_seq.append(str(action[0]))
                    break

                # Normal progression, reward comes from previous step
                state_seq.append({
                    "state": traj[j]['anchor_obs'],
                    "reward": traj[j-1]['rewards'],
                    "active_masks": traj[j-1]['active_masks']
                })
                raw_response = tokenizer.decode(traj[j-1]['responses'])
                action, valid = self.projection_f([raw_response])
                action_seq.append(str(action[0]))

            raw_state_list.append(state_seq)
            raw_action_list.append(action_seq)

        return raw_state_list, raw_action_list

    def clean_trajectory(self, trajectory, idx_to_state):
        cleaned_trajectory = []
        for i in range(len(trajectory)):
            cleaned_trajectory.append([])
            for triple in trajectory[i]:
                if triple[0] != triple[2]:
                    cleaned_trajectory[i].append(triple)
        return cleaned_trajectory



class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

    def state_preprocess(self, total_batch_list, tokenizer):
        def extract_actions(action):
            pattern = r"click\[(.*)\]"

            match = re.search(pattern, action)
            if match:
                return match.group(1)
            return action

        # attributes_manager需要固定顺序，确保不出现冗余状态
        raw_state_list = []
        raw_action_list = []

        for traj in total_batch_list:
            # Initialize state and action sequences for each trajectory
            state_seq = [{
                "state": traj[0]['anchor_obs'],
                "reward": 0,
                "active_masks": traj[0]['active_masks']
            }]
            action_seq = []

            # 使用ordered dict以固定顺序存储attributes
            attributes_manager = collections.OrderedDict()
            for j in range(1, len(traj)):
                # If current step's active_masks is False, trajectory ends
                if not traj[j]['active_masks']:
                    # If not the last step (success), add the last valid state and action
                    state_seq.append({
                        "state": traj[j]['anchor_obs'],
                        "reward": traj[j-1]['rewards'],
                        "active_masks": traj[j-1]['active_masks']
                    })
                    raw_response = tokenizer.decode(traj[j-1]['responses'])
                    action, valid = self.projection_f([raw_response])
                    action_seq.append(str(action[0]))
                    break

                state = traj[j]['anchor_obs']
                raw_response = tokenizer.decode(traj[j-1]['responses'])
                action, valid = self.projection_f([raw_response])
                extracted_action = extract_actions(action[0])
                info = traj[j-1]['infos']
                action_space = traj[j-1]['infos']["available_actions"]['clickables']

                # 修改goods_attributes_manager
                if info["available_actions"]['has_search_bar']:
                    # 搜索页 & 完成页： 清空attributes_manager
                    attributes_manager.clear()
                elif "buy now" not in action_space:
                    # 召回页：清空attributes_manager
                    attributes_manager.clear()
                elif "buy now" in action_space:
                    # 商详页：为商品添加选择的属性
                    attributes_list = [a for a in action_space if a not in ['back to search', '< prev', 'description', 'features', 'reviews', 'buy now']]
                    if extracted_action in attributes_list:
                        if extracted_action not in attributes_manager:
                            attributes_manager[extracted_action] = None

                if len(attributes_manager) > 0:
                    # 固定顺序，避免冗余状态
                    clicked_description = f" clicked attributes: [{', '.join(attributes_manager.keys())}]"
                else:
                    clicked_description = ""

                # Normal progression, reward comes from previous step
                state_seq.append({
                    "state": traj[j]['anchor_obs'] + clicked_description,
                    "reward": traj[j-1]['rewards'],
                    "active_masks": traj[j-1]['active_masks']
                })
                
                action_seq.append(str(action[0]))

            raw_state_list.append(state_seq)
            raw_action_list.append(action_seq)

        print("end webshop state_preprocess")
        return raw_state_list, raw_action_list
  
    def clean_trajectory(self, trajectory, idx_to_state):
        cleaned_trajectory = []
        for i in range(len(trajectory)):
            cleaned_trajectory.append([])
            for triple in trajectory[i]:
                if triple[0] != triple[2]:
                    cleaned_trajectory[i].append(triple)
        return cleaned_trajectory

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
