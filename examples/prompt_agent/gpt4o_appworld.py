import os
import numpy as np
import time
import logging
from datetime import datetime
from collections import defaultdict
from openai import OpenAI
import json

# Import AppWorld prompt templates
from agent_system.environments.prompts.appworld import APPWORLD_TEMPLATE_NO_HIS, APPWORLD_TEMPLATE

# Import AppWorld environment builder
from agent_system.environments.env_package.appworld import build_appworld_envs


def extract_code_from_response(response: str) -> str:
    """
    Extract the code from <code>...</code> tags in the response.
    If not found, return the last 100 characters as fallback.
    """
    start_tag = "<code>"
    end_tag = "</code>"
    start_idx = response.find(start_tag)
    end_idx = response.find(end_tag)
    
    if start_idx != -1 and end_idx != -1:
        return response[start_idx + len(start_tag):end_idx].strip()
    else:
        # Fallback: return last 100 characters
        return response[-100:].strip()


class SimpleMemory:
    """Simple memory to store action history for each environment."""
    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size
        self.memory = [[] for _ in range(batch_size)]
    
    def reset(self, batch_size: int):
        self.batch_size = batch_size
        self.memory = [[] for _ in range(batch_size)]
    
    def store(self, idx: int, record: dict):
        self.memory[idx].append(record)
    
    def __getitem__(self, idx: int):
        return self.memory[idx]


class AppWorldAgent:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key="123456",
            base_url="http://localhost:8000/v1",
        )
        
    def get_action_from_gpt(self, prompt: str) -> str:
        """Get action from GPT model."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.4,
            n=1,
            stop=None
        )
        action = response.choices[0].message.content.strip()
        return action


def build_initial_prompt(task: str, supervisor: dict) -> str:
    """Build the initial prompt with no history."""
    return APPWORLD_TEMPLATE_NO_HIS.format(
        supervisor_first_name=supervisor['first_name'],
        supervisor_last_name=supervisor['last_name'],
        supervisor_email=supervisor['email'],
        supervisor_phone_number=supervisor['phone_number'],
        task_description=task,
    )


def build_prompt_with_history(task: str, supervisor: dict, memory: list, history_length: int = 10) -> str:
    """Build the prompt with action history."""
    # Get last `history_length` steps
    recent_history = memory[-history_length:]
    valid_history_length = len(recent_history)
    start_index = len(memory) - valid_history_length
    
    action_history = ""
    for j, record in enumerate(recent_history):
        step_number = start_index + j + 1
        action = record["action"]
        env_obs = record["text_obs"]
        action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
    
    # Truncate if too long
    if len(action_history) > 10000:
        action_history = "... " + action_history[-10000:]
    
    return APPWORLD_TEMPLATE.format(
        supervisor_first_name=supervisor['first_name'],
        supervisor_last_name=supervisor['last_name'],
        supervisor_email=supervisor['email'],
        supervisor_phone_number=supervisor['phone_number'],
        task_description=task,
        step_count=len(memory),
        history_length=valid_history_length,
        action_history=action_history.strip(),
        current_step=len(memory) + 1,
    )


if __name__ == "__main__":

    # -------- logging ----------
    os.makedirs("logs/appworld", exist_ok=True)
    log_fp = os.path.join(
        "logs/appworld", f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    # -------- Parameters ----------
    max_steps = 50
    env_num = 50  # Number of environments to run in parallel
    test_times = 3  # Number of test rounds
    dataset_name = "test_normal"  # 'train', 'test_normal', or 'test_challenge'
    history_length = 5  # Number of recent history steps to include in prompt
    port_file = "appworld_ports.ports"  # Port file for AppWorld servers

    # -------- Environment and agent setup ----------
    logging.info(f"Building AppWorld environments with dataset: {dataset_name}, env_num: {env_num}")
    
    envs = build_appworld_envs(
        dataset_name=dataset_name,
        max_interactions=max_steps,
        seed=42,
        env_num=env_num,
        group_n=1,
        start_server_id=0,
        resources_per_worker={"num_cpus": 0.05, "num_gpus": 0.0},
    )
    
    agent = AppWorldAgent(model_name="/mnt/data1/data/models/Qwen2.5-Coder-32B-Instruct")
    memory = SimpleMemory(batch_size=env_num)

    # Accumulated statistics
    overall_success_rates = []

    # ======================= Main Loop =======================
    for test_idx in range(test_times):
        logging.info(f"\n========== Start test {test_idx} ==========")
        start_time = time.time()
        
        # Reset environments
        obs_list, info_list = envs.reset()
        memory.reset(batch_size=env_num)
        
        # Store task and supervisor information
        tasks = obs_list.copy()
        supervisors = [info['supervisor'] for info in info_list]
        task_ids = [info['task_id'] for info in info_list]
        
        env_dones = [False] * env_num
        overall_success_this_round = np.zeros(env_num, dtype=bool)
        record = [[] for _ in range(env_num)]
        success_record = [False] * env_num

        for step_idx in range(max_steps):
            logging.info(f"Step {step_idx}; Dones ({np.array(env_dones).sum().item()}/{env_num}); SR {overall_success_this_round.mean().item():.4f}")

            # --- Build prompts and get actions ---
            actions = []
            raw_responses = []
            
            for i in range(env_num):
                if env_dones[i]:
                    actions.append("None")
                    raw_responses.append("None")
                else:
                    # Build prompt
                    if step_idx == 0:
                        prompt = build_initial_prompt(tasks[i], supervisors[i])
                    else:
                        prompt = build_prompt_with_history(tasks[i], supervisors[i], memory[i], history_length)
                    
                    # Get action from GPT
                    raw_response = agent.get_action_from_gpt(prompt)
                    raw_responses.append(raw_response)
                    
                    # Extract code from response
                    code = extract_code_from_response(raw_response)
                    actions.append(code)
                    
                    # Store record
                    record[i].append({"raw_response": raw_response, "action": code})

            # --- Environment stepping ---
            obs_list, rewards, dones, infos = envs.step(actions)
            for i in range(env_num):
                record[i][-1]["obs"] = obs_list[i]
                record[i][-1]["reward"] = rewards[i]
                record[i][-1]["done"] = dones[i]
                # record[i][-1]["info"] = infos[i]

            # --- Store history and determine endings ---
            for i in range(env_num):
                if env_dones[i]:
                    continue
                
                # Store action and observation in memory
                memory.store(i, {
                    "action": actions[i],
                    "text_obs": obs_list[i],
                })

                if dones[i]:
                    env_dones[i] = True
                    won = bool(infos[i].get("won", False))
                    overall_success_this_round[i] = won
                    success_record[i] = won
                    
                    logging.info(f"  Env {i} (task_id={task_ids[i]}) finished: {'SUCCESS' if won else 'FAIL'}")

            if all(env_dones):
                logging.info("All environments finished early!")
                break

        # -------- Save record to JSON --------
        full_record = [
            {
                "model_name": agent.model_name,
                "task_id": task_ids[i],
                "task": tasks[i],
                "success": success_record[i],
                "record": record[i],
            } for i in range(env_num)
        ]
        with open(f"logs/appworld/record_{agent.model_name.split('/')[-1]}_test_{test_idx}.json", "w") as f:
            json.dump(full_record, f, indent=4)

        # -------- Single round results --------
        round_success_rate = overall_success_this_round.mean()
        overall_success_rates.append(round_success_rate)

        logging.info(f"Test {test_idx} overall success: {round_success_rate:.4f}")
        logging.info(
            f"Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n"
        )

    # ======================= Final Summary =======================
    logging.info("=============== Final Summary ===============")
    logging.info(
        f"Total tests: {test_times} | Envs / test: {env_num} | Total envs: {env_num * test_times}"
    )
    logging.info(
        f"Overall success avg ± std: "
        f"{np.mean(overall_success_rates):.4f} ± {np.std(overall_success_rates):.4f}"
    )

    # Close environments
    envs.close()

