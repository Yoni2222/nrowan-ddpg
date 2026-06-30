"""
NROWAN-DDPG vs Vanilla DDPG on a purpose-built DECEPTIVE CONTINUOUS task.

This environment isolates the exact phenomenon NROWAN's coherent parameter-
space noise is designed for: escaping a deceptive local optimum that a
gradient-following agent with uncorrelated (per-step Gaussian) noise gets
stuck in.

The task (1-D corridor):
  * The agent starts at x = 0 and moves with a continuous action a in [-1, 1]
    (x <- clip(x + step*a, -1, 1)).
  * A small DECEPTIVE reward bump sits at x = -0.5 (the "trap"): the immediate
    reward gradient points LEFT.
  * The real GOAL is far to the RIGHT (x >= 0.9), worth much more, but reaching
    it requires crossing a "desert" of (near-)zero reward.

Expected outcome:
  * Vanilla DDPG follows the gradient into the trap; its per-step Gaussian noise
    averages to ~0 net displacement, so it cannot sustain the rightward push
    needed to cross the desert -> never reaches the goal.
  * NROWAN-DDPG's coherent per-episode parameter noise gives a consistent
    directional drift; some episodes cross the desert, discover the goal, and
    the agent learns to go for it.

The agent / network / noise code is reused UNCHANGED; only the env differs.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from agent.ddpg_agent import DDPGAgent
from agent.memory import ReplayBuffer


class DeceptiveCorridorEnv:
    """Minimal gymnasium-style 1-D deceptive corridor (see module docstring)."""

    def __init__(self, horizon=120, step_size=0.03,
                 trap_x=-0.5, trap_w=0.15, trap_r=0.05,
                 goal_x=0.9, goal_r=10.0, time_cost=0.01):
        self.horizon = horizon
        self.step_size = step_size
        self.trap_x, self.trap_w, self.trap_r = trap_x, trap_w, trap_r
        self.goal_x, self.goal_r = goal_x, goal_r
        self.time_cost = time_cost
        self.x, self.t = 0.0, 0

    def reset(self, seed=None):
        self.x, self.t = 0.0, 0
        return np.array([self.x], dtype=np.float32), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).flatten()[0], -1.0, 1.0))
        self.x = float(np.clip(self.x + self.step_size * a, -1.0, 1.0))
        self.t += 1

        terminated = self.x >= self.goal_x
        truncated = self.t >= self.horizon
        if terminated:
            reward = self.goal_r
        else:
            # deceptive bump (positive, pulls the greedy agent left) minus a
            # small per-step time cost (so loitering in the trap is sub-optimal)
            trap = self.trap_r * np.exp(-((self.x - self.trap_x) ** 2) /
                                        (2.0 * self.trap_w ** 2))
            reward = trap - self.time_cost
        return np.array([self.x], dtype=np.float32), float(reward), terminated, truncated, {}


def moving_average(data, window):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def run_training(mode, seed, n_episodes, warmup_steps, batch_size,
                 sigma_init, xi_max):
    """Train one agent (mode='nrowan' or 'vanilla') and return per-episode
    metrics: true reward, solved (reached goal), final x position."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = DeceptiveCorridorEnv()
    state_dim, action_dim, max_action = 1, 1, 1.0
    agent = DDPGAgent(state_dim, action_dim, max_action,
                      sigma_init=sigma_init, xi_max=xi_max, mode=mode)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    ep_rewards, ep_solved, ep_finalx = [], [], []
    total_steps = 0

    for episode in tqdm(range(n_episodes), desc=f"{mode:7s} seed={seed}"):
        state, _ = env.reset(seed=seed + episode)
        state = np.asarray(state, dtype=np.float32)

        ep_reward, solved = 0.0, 0
        agent.reset_exploration_noise()   # NROWAN: coherent per-episode noise

        while True:
            if total_steps < warmup_steps:
                action = np.random.uniform(-max_action, max_action,
                                           size=action_dim).astype(np.float32)
            else:
                action = agent.select_action(state, explore=True).astype(np.float32)

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.asarray(next_state, dtype=np.float32)
            done = bool(terminated)

            replay_buffer.add(state, action, reward, next_state, float(done))

            if total_steps >= warmup_steps and replay_buffer.size > batch_size:
                agent.train(replay_buffer, batch_size)

            state = next_state
            ep_reward += reward
            total_steps += 1

            if terminated:
                solved = 1
            if terminated or truncated:
                break

        ep_rewards.append(ep_reward)
        ep_solved.append(solved)
        ep_finalx.append(float(state[0]))
        # xi gated on actual success (keeps exploration ON until the goal is found)
        agent.update_noise_weight(float(solved))

    return agent, {"rewards": ep_rewards, "solved": ep_solved, "finalx": ep_finalx}


def plot_comparison(agg, results_dir, ma_window):
    colors = {"nrowan": "green", "vanilla": "darkorange"}
    labels = {"nrowan": "NROWAN-DDPG (ours)", "vanilla": "Vanilla DDPG"}

    plt.figure(figsize=(14, 5))

    # --- Episode reward --- #
    plt.subplot(1, 2, 1)
    for mode, data in agg.items():
        rewards = data["rewards"]
        mean = rewards.mean(axis=0)
        ep = np.arange(1, len(mean) + 1)
        for s in range(rewards.shape[0]):
            plt.plot(ep, rewards[s], color=colors[mode], alpha=0.10)
        ma = moving_average(mean, ma_window)
        plt.plot(np.arange(ma_window, len(mean) + 1), ma,
                 color=colors[mode], linewidth=2.5, label=labels[mode])
    plt.axhline(10, color='gray', linestyle='--', linewidth=1.5,
                label='goal reward (+10)')
    plt.title('Episode reward (higher = better)')
    plt.xlabel('Episode'); plt.ylabel('Total reward')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    # --- Success rate --- #
    plt.subplot(1, 2, 2)
    for mode, data in agg.items():
        solved = data["solved"]
        mean = solved.mean(axis=0)
        ma = moving_average(mean, ma_window)
        plt.plot(np.arange(ma_window, len(mean) + 1), ma,
                 color=colors[mode], linewidth=2.5, label=labels[mode])
    plt.title(f'Success rate (escaped trap -> reached goal), {ma_window}-ep MA')
    plt.xlabel('Episode'); plt.ylabel('Fraction of episodes solved')
    plt.ylim(-0.05, 1.05)
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_deceptive.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"=> Comparison graph saved to: {path}")


def main():
    results_dir = "results_deceptive"
    os.makedirs(results_dir, exist_ok=True)

    MAX_EPISODES = 150
    BATCH_SIZE = 128
    WARMUP_STEPS = 1000
    SIGMA_INIT = 1.0
    XI_MAX = 0.5
    SEEDS = [0, 1, 2]
    MA_WINDOW = 10

    agg = {}
    for mode in ["nrowan", "vanilla"]:
        per_seed = {"rewards": [], "solved": [], "finalx": []}
        for seed in SEEDS:
            print(f"\n=== Training [{mode}] seed={seed} for {MAX_EPISODES} episodes ===")
            _, res = run_training(mode, seed, MAX_EPISODES, WARMUP_STEPS,
                                  BATCH_SIZE, SIGMA_INIT, XI_MAX)
            for k in per_seed:
                per_seed[k].append(res[k])
        agg[mode] = {k: np.array(v, dtype=float) for k, v in per_seed.items()}
        np.savetxt(os.path.join(results_dir, f"rewards_{mode}.txt"), agg[mode]["rewards"])
        np.savetxt(os.path.join(results_dir, f"solved_{mode}.txt"), agg[mode]["solved"])

    plot_comparison(agg, results_dir, MA_WINDOW)

    print("\n========= SUMMARY (last-30-episode averages over seeds) =========")
    print(f"{'method':18s} {'reward':>10s} {'success%':>10s} {'final_x':>10s}")
    for mode in ["nrowan", "vanilla"]:
        r = agg[mode]["rewards"][:, -30:].mean()
        s = agg[mode]["solved"][:, -30:].mean() * 100
        fx = agg[mode]["finalx"][:, -30:].mean()
        print(f"{mode:18s} {r:10.2f} {s:10.1f} {fx:10.2f}")
    print("=================================================================")


if __name__ == "__main__":
    main()
