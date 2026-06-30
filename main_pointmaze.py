"""
NROWAN-DDPG vs Vanilla DDPG on PointMaze (gymnasium-robotics), sparse reward.

PointMaze is a recognized continuous-control benchmark: a point mass must
navigate a maze (continuous 2-D force action) to a fixed goal. With the SPARSE
reward (reward = 1 only on reaching the goal, 0 otherwise) it is a genuine
hard-exploration task: there is no signal guiding the agent until it first
reaches the goal, and the maze walls force it to commit to a route rather than
move straight toward the target. We use the Medium maze (a multi-corridor grid
with dead-ends), where reaching the goal requires committing to a specific
multi-turn route over many consecutive steps -- precisely the regime where
coherent per-episode noise should beat per-step jitter.

This is the regime where NROWAN's coherent per-episode parameter noise is
expected to beat vanilla DDPG's per-step Gaussian action noise: committed
exploration can navigate around the wall, whereas jitter that averages to ~0
cannot sustain the detour.

The agent / network / noise code is reused UNCHANGED; only the env loop differs.
Runs 3 seeds per method and reports mean +/- standard deviation.
"""
import os
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
from tqdm import tqdm

from agent.ddpg_agent import DDPGAgent
from agent.memory import ReplayBuffer

gym.register_envs(gymnasium_robotics)

ENV_ID = "PointMaze_Medium-v3"
MAX_STEPS = 600


def make_env():
    # fixed goal (reset_target=False) + terminate on reaching it (continuing_task=False)
    # => a clean single-goal hard-exploration navigation task.
    return gym.make(ENV_ID, max_episode_steps=MAX_STEPS,
                    continuing_task=False, reset_target=False)


def flatten_obs(obs):
    """Dict observation -> flat state vector [position, velocity, goal] (6-D)."""
    return np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)


def moving_average(data, window):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def run_training(mode, seed, state_dim, action_dim, max_action,
                 n_episodes, warmup_steps, batch_size, sigma_init, xi_max):
    """Train one agent (mode='nrowan' or 'vanilla') on a single seed.
    Returns per-episode metrics: reward, solved (reached goal), length (steps)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env()
    agent = DDPGAgent(state_dim, action_dim, max_action,
                      sigma_init=sigma_init, xi_max=xi_max, mode=mode)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    ep_rewards, ep_solved, ep_lengths = [], [], []
    total_steps = 0

    for episode in tqdm(range(n_episodes), desc=f"{mode:7s} seed={seed}"):
        obs, _ = env.reset(seed=seed + episode)
        state = flatten_obs(obs)

        ep_reward, solved, length = 0.0, 0, 0
        agent.reset_exploration_noise()   # NROWAN: coherent per-episode noise

        while True:
            if total_steps < warmup_steps:
                action = np.random.uniform(-max_action, max_action,
                                           size=action_dim).astype(np.float32)
            else:
                action = agent.select_action(state, explore=True).astype(np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = flatten_obs(next_obs)
            done = bool(terminated)   # only a REAL terminal (goal) zeroes the bootstrap

            replay_buffer.add(state, action, reward, next_state, float(done))

            if total_steps >= warmup_steps and replay_buffer.size > batch_size:
                agent.train(replay_buffer, batch_size)

            state = next_state
            ep_reward += reward
            length += 1
            total_steps += 1

            if terminated or bool(info.get("success", False)):
                solved = 1
            if terminated or truncated:
                break

        ep_rewards.append(ep_reward)
        ep_solved.append(solved)
        ep_lengths.append(length)
        # xi gated on success (keeps exploration ON until the goal is found)
        agent.update_noise_weight(float(solved))

    env.close()
    return agent, {"rewards": ep_rewards, "solved": ep_solved, "lengths": ep_lengths}


def plot_comparison(agg, results_dir, ma_window):
    """agg[mode][key] is a 2D array [n_seeds, n_episodes]. Plots mean +/- std band."""
    colors = {"nrowan": "green", "vanilla": "darkorange"}
    labels = {"nrowan": "NROWAN-DDPG (ours)", "vanilla": "Vanilla DDPG"}

    def smoothed_mean_std(arr):
        # smooth each seed, then take mean/std across seeds
        sm = np.array([moving_average(arr[s], ma_window) for s in range(arr.shape[0])])
        return sm.mean(axis=0), sm.std(axis=0)

    plt.figure(figsize=(14, 5))

    # --- Success rate --- #
    plt.subplot(1, 2, 1)
    for mode, data in agg.items():
        mean, std = smoothed_mean_std(data["solved"])
        x = np.arange(ma_window, ma_window + len(mean))
        plt.plot(x, mean, color=colors[mode], linewidth=2.5, label=labels[mode])
        plt.fill_between(x, mean - std, mean + std, color=colors[mode], alpha=0.20)
    plt.title(f'Success rate (reached goal), {ma_window}-ep MA  [mean $\\pm$ std, 3 seeds]')
    plt.xlabel('Episode'); plt.ylabel('Fraction of episodes solved')
    plt.ylim(-0.05, 1.05)
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    # --- Steps to goal (lower = better) --- #
    plt.subplot(1, 2, 2)
    for mode, data in agg.items():
        mean, std = smoothed_mean_std(data["lengths"])
        x = np.arange(ma_window, ma_window + len(mean))
        plt.plot(x, mean, color=colors[mode], linewidth=2.5, label=labels[mode])
        plt.fill_between(x, mean - std, mean + std, color=colors[mode], alpha=0.20)
    plt.title(f'Episode length (steps; lower once solving)  [mean $\\pm$ std]')
    plt.xlabel('Episode'); plt.ylabel('Steps')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_pointmaze.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"=> Comparison graph saved to: {path}")


def main():
    results_dir = "results_pointmaze"
    os.makedirs(results_dir, exist_ok=True)

    # discover dims from the env
    env = make_env()
    dummy, _ = env.reset(seed=0)
    state_dim = flatten_obs(dummy).shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    env.close()
    print(f"state_dim={state_dim}  action_dim={action_dim}  max_action={max_action}")

    MAX_EPISODES = 200
    BATCH_SIZE = 128
    WARMUP_STEPS = 1000
    SIGMA_INIT = 0.5
    XI_MAX = 0.5
    SEEDS = [0, 1, 2]
    MA_WINDOW = 10

    agg = {}
    for mode in ["nrowan", "vanilla"]:
        per_seed = {"rewards": [], "solved": [], "lengths": []}
        for seed in SEEDS:
            print(f"\n=== Training [{mode}] seed={seed} for {MAX_EPISODES} episodes ===")
            _, res = run_training(mode, seed, state_dim, action_dim, max_action,
                                  MAX_EPISODES, WARMUP_STEPS, BATCH_SIZE, SIGMA_INIT, XI_MAX)
            for k in per_seed:
                per_seed[k].append(res[k])
        agg[mode] = {k: np.array(v, dtype=float) for k, v in per_seed.items()}
        for k in ["rewards", "solved", "lengths"]:
            np.savetxt(os.path.join(results_dir, f"{k}_{mode}.txt"), agg[mode][k])

    plot_comparison(agg, results_dir, MA_WINDOW)

    # --- Final verdict: mean +/- std across the 3 seeds (last-30-episode averages) --- #
    print("\n========= SUMMARY (last-30-ep averages: mean +/- std over 3 seeds) =========")
    print(f"{'method':18s} {'success%':>16s} {'steps':>16s}")
    for mode in ["nrowan", "vanilla"]:
        s_per_seed = agg[mode]["solved"][:, -30:].mean(axis=1) * 100   # one number per seed
        l_per_seed = agg[mode]["lengths"][:, -30:].mean(axis=1)
        print(f"{mode:18s} {s_per_seed.mean():7.1f} +/- {s_per_seed.std():5.1f}    "
              f"{l_per_seed.mean():7.1f} +/- {l_per_seed.std():5.1f}")
    print("============================================================================")


if __name__ == "__main__":
    main()
