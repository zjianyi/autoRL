"""
Fixed evaluation harness for autoRL BipedalWalker experiments.
DO NOT MODIFY — this file defines the ground-truth metric.

Imported by train.py:
    from evaluate import TIME_BUDGET, ENV_ID, make_env, evaluate_policy
"""

import numpy as np
import gymnasium as gym

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 720          # wall-clock training time budget in seconds (12 minutes)
ENV_ID = "BipedalWalker-v3"  # environment — ~300 is "solved", negative scores are common early
NUM_EVAL_EPISODES = 100    # number of episodes used for final evaluation

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(env_id, idx, seed=1):
    """Return a thunk that creates and seeds a single gym environment."""
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env
    return thunk

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_policy(agent, device, env_id=ENV_ID, num_episodes=NUM_EVAL_EPISODES):
    """
    Run the agent deterministically for num_episodes episodes and return summary stats.

    agent must implement:
        agent.get_action(obs_tensor) -> np.ndarray   (deterministic / mean action)

    Returns a dict with keys:
        avg_return, std_return, min_return, max_return
    """
    import torch
    env = gym.make(env_id)
    returns = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_return = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent.get_action(obs_t)  # returns np.ndarray
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)

    env.close()
    returns = np.array(returns)
    return {
        "avg_return": float(np.mean(returns)),
        "std_return":  float(np.std(returns)),
        "min_return":  float(np.min(returns)),
        "max_return":  float(np.max(returns)),
    }
