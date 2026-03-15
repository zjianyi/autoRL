"""
autoRL PPO training script for BipedalWalker-v3. Single-file, single-process.
Usage: python train.py

The agent edits ONLY this file. evaluate.py is read-only.
"""

import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions.normal import Normal

from evaluate import TIME_BUDGET, ENV_ID, make_env, evaluate_policy

# ---------------------------------------------------------------------------
# Hyperparameters — edit these freely
# ---------------------------------------------------------------------------

SEED = 1
LEARNING_RATE = 2.5e-4
NUM_ENVS = 4              # number of parallel environments
NUM_STEPS = 128           # rollout steps per environment per update
GAMMA = 0.99              # discount factor
GAE_LAMBDA = 0.95         # GAE lambda
NUM_MINIBATCHES = 4
UPDATE_EPOCHS = 4         # PPO update epochs per rollout
NORM_ADV = True           # normalize advantages
CLIP_COEF = 0.2           # PPO clip coefficient
CLIP_VLOSS = True         # clip value loss
ENT_COEF = 0.01           # entropy bonus coefficient
VF_COEF = 0.5             # value loss coefficient
MAX_GRAD_NORM = 0.5       # gradient clip norm
TARGET_KL = None          # optional early-stop threshold on approx KL
ANNEAL_LR = True          # linearly anneal LR to 0 over training

# ---------------------------------------------------------------------------
# Agent — edit architecture freely
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)),      nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)),      nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        log_std = torch.clamp(self.actor_log_std, -5, 2).expand_as(mean)
        std = log_std.exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, self.critic(x)

    def get_action(self, x):
        """Deterministic (mean) action for evaluation. Returns np.ndarray."""
        with torch.no_grad():
            return self.actor_mean(x).squeeze(0).cpu().numpy()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = NUM_ENVS * NUM_STEPS
minibatch_size = batch_size // NUM_MINIBATCHES

envs = gym.vector.SyncVectorEnv(
    [make_env(ENV_ID, i, seed=SEED) for i in range(NUM_ENVS)]
)
assert isinstance(envs.single_action_space, gym.spaces.Box)

agent = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

# Rollout buffers
obs      = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(device)
actions  = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
rewards  = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
dones    = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
values   = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)

# ---------------------------------------------------------------------------
# Training loop — runs for TIME_BUDGET seconds
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0.0
global_step = 0
update = 0

next_obs_np, _ = envs.reset(seed=SEED)
next_obs  = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
next_done = torch.zeros(NUM_ENVS).to(device)

while True:
    t0 = time.time()

    if ANNEAL_LR:
        frac = max(0.0, 1.0 - total_training_time / TIME_BUDGET)
        optimizer.param_groups[0]["lr"] = frac * LEARNING_RATE

    # --- Rollout ---
    for step in range(NUM_STEPS):
        global_step += NUM_ENVS
        obs[step]   = next_obs
        dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step]  = action
        logprobs[step] = logprob

        action_np = action.cpu().numpy()
        action_np = np.clip(action_np, envs.single_action_space.low, envs.single_action_space.high)

        next_obs_np, reward, terminations, truncations, infos = envs.step(action_np)
        next_done_np = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device)
        next_obs  = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
        next_done = torch.tensor(next_done_np, dtype=torch.float32).to(device)

    # --- GAE advantage estimation ---
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(NUM_STEPS)):
            if t == NUM_STEPS - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
        returns = advantages + values

    # --- Flatten batch ---
    b_obs        = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs   = logprobs.reshape(-1)
    b_actions    = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns    = returns.reshape(-1)
    b_values     = values.reshape(-1)

    # --- PPO update ---
    b_inds = np.arange(batch_size)
    for epoch in range(UPDATE_EPOCHS):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            mb_inds = b_inds[start:start + minibatch_size]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds]
            )
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()

            mb_adv = b_advantages[mb_inds]
            if NORM_ADV:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            pg_loss = torch.max(
                -mb_adv * ratio,
                -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF),
            ).mean()

            newvalue = newvalue.view(-1)
            if CLIP_VLOSS:
                v_loss = 0.5 * torch.max(
                    (newvalue - b_returns[mb_inds]) ** 2,
                    (b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -CLIP_COEF, CLIP_COEF
                    ) - b_returns[mb_inds]) ** 2,
                ).mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        if TARGET_KL is not None and approx_kl > TARGET_KL:
            break

    dt = time.time() - t0
    total_training_time += dt
    update += 1

    print(
        f"\rupdate {update:05d} | steps {global_step:,} | "
        f"pg_loss: {pg_loss.item():.4f} | v_loss: {v_loss.item():.4f} | "
        f"entropy: {entropy.mean().item():.4f} | "
        f"remaining: {max(0, TIME_BUDGET - total_training_time):.0f}s    ",
        end="", flush=True,
    )

    if total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r log

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

agent.eval()
stats = evaluate_policy(agent, device)

t_end = time.time()
print("---")
print(f"avg_return:       {stats['avg_return']:.6f}")
print(f"std_return:       {stats['std_return']:.6f}")
print(f"min_return:       {stats['min_return']:.6f}")
print(f"max_return:       {stats['max_return']:.6f}")
print(f"total_timesteps:  {global_step}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"num_updates:      {update}")

envs.close()
