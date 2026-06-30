"""
NROWAN-DDPG vs Vanilla DDPG on MountainCarContinuous-v0.

This is the hard-exploration continuous-control benchmark where NROWAN's
learned parameter-space noise is expected to clearly beat vanilla DDPG's
external Gaussian action noise (cf. Plappert et al., 2017).

Why this env: the reward is sparse (+100 only on reaching the goal, minus a
small -0.1*a^2 control cost every step). Vanilla DDPG tends to converge to the
do-nothing action (output 0) to avoid the control penalty and NEVER discovers
the "swing left first to build momentum" strategy -> reward stays ~0. NROWAN's
coherent per-episode parameter noise drives consistent, directed exploration
that discovers the goal.

The agent/network/noise code is reused UNCHANGED from the grid2op experiment;
only the environment loop differs.
"""
import os
import math
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

DISCOUNT = 0.99   # must match the agent's discount for potential-based shaping


def potential(state):
    """Potential Phi(s) for reward shaping = a mechanical-energy proxy:
    height (sin(3*position)) + scaled kinetic energy (velocity^2). Potential-
    based shaping F = gamma*Phi(s') - Phi(s) provably preserves the optimal
    policy (Ng et al., 1999), so it guides learning toward building momentum
    WITHOUT changing the underlying task. Applied identically to both methods."""
    p, v = float(state[0]), float(state[1])
    return math.sin(3.0 * p) + 100.0 * (v ** 2)

from agent.ddpg_agent import DDPGAgent
from agent.memory import ReplayBuffer


def get_save_paths():
    colab_drive_path = '/content/drive/MyDrive/'
    if os.path.exists(colab_drive_path):
        print("Google Drive detection: SUCCESS. Training data will sync to cloud.")
        base_dir = os.path.join(colab_drive_path, 'NROWAN_DDPG_Project')
        models_dir = os.path.join(base_dir, 'saved_models_mc')
        results_dir = os.path.join(base_dir, 'results_mc')
    else:
        print("Google Drive detection: NOT FOUND. Saving to local project directory...")
        models_dir = "saved_models_mc"
        results_dir = "results_mc"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return models_dir, results_dir


def moving_average(data, window):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def run_training(mode, seed, state_dim, action_dim, max_action,
                 n_episodes, warmup_steps, batch_size, sigma_init, xi_max,
                 use_shaping=True):
    """Train one agent (mode='nrowan' or 'vanilla') on a single seed and return
    per-episode metrics: reward, solved (reached goal), length (steps)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("MountainCarContinuous-v0")
    agent = DDPGAgent(state_dim, action_dim, max_action,
                      sigma_init=sigma_init, xi_max=xi_max, mode=mode)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    ep_rewards, ep_solved, ep_lengths, ep_sigma = [], [], [], []
    total_steps = 0

    for episode in tqdm(range(n_episodes), desc=f"{mode:7s} seed={seed}"):
        state, _ = env.reset(seed=seed + episode)
        state = np.asarray(state, dtype=np.float32)

        ep_reward_true, ep_reward_shaped, length, solved = 0.0, 0.0, 0, 0
        agent.reset_exploration_noise()   # NROWAN: coherent per-episode noise

        while True:
            if total_steps < warmup_steps:
                action = np.random.uniform(
                    -max_action, max_action, size=action_dim).astype(np.float32)
            else:
                action = agent.select_action(state, explore=True).astype(np.float32)

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.asarray(next_state, dtype=np.float32)
            done = bool(terminated)  # only a REAL terminal (goal) zeroes the bootstrap

            # Potential-based shaping (identical for both methods): the agent
            # LEARNS from the shaped reward (dense momentum-building gradient),
            # but we REPORT the true env reward + success rate, which shaping
            # cannot fake -> the comparison stays honest. With use_shaping=False
            # the task is PURE SPARSE (hard exploration) where NROWAN's coherent
            # noise is expected to beat vanilla's Gaussian noise.
            if use_shaping:
                shaped = reward + DISCOUNT * potential(next_state) - potential(state)
            else:
                shaped = reward

            replay_buffer.add(state, action, shaped, next_state, float(done))

            if total_steps >= warmup_steps and replay_buffer.size > batch_size:
                agent.train(replay_buffer, batch_size)

            state = next_state
            ep_reward_true += reward
            ep_reward_shaped += shaped
            length += 1
            total_steps += 1

            if terminated:
                solved = 1
            if terminated or truncated:
                break

        ep_rewards.append(ep_reward_true)     # true env reward for honest reporting
        ep_solved.append(solved)
        ep_lengths.append(length)
        ep_sigma.append(agent.noise_magnitude())   # output-layer sigma diagnostic
        # NROWAN online weight adjustment is driven by SUCCESS (reaching the goal),
        # not raw reward: in the sparse task "high reward" pre-solve = do-nothing,
        # so gating xi on actual success keeps exploration ON until the agent can
        # truly solve, then anneals the noise. (Vanilla ignores this arg.)
        agent.update_noise_weight(float(solved))

    env.close()
    return agent, {"rewards": ep_rewards, "solved": ep_solved,
                   "lengths": ep_lengths, "sigma": ep_sigma}


def plot_comparison(agg, results_dir, ma_window):
    """agg[mode][key] is a 2D array [n_seeds, n_episodes]."""
    colors = {"nrowan": "green", "vanilla": "darkorange"}
    labels = {"nrowan": "NROWAN-DDPG (ours)", "vanilla": "Vanilla DDPG"}

    plt.figure(figsize=(14, 5))

    # --- Episode reward (headline) --- #
    plt.subplot(1, 2, 1)
    for mode, data in agg.items():
        rewards = data["rewards"]                 # [seeds, episodes]
        mean = rewards.mean(axis=0)
        ep = np.arange(1, len(mean) + 1)
        ma = moving_average(mean, ma_window)
        ma_x = np.arange(ma_window, len(mean) + 1)
        # per-seed faint lines for transparency
        for s in range(rewards.shape[0]):
            plt.plot(ep, rewards[s], color=colors[mode], alpha=0.10)
        plt.plot(ma_x, ma, color=colors[mode], linewidth=2.5, label=labels[mode])
    plt.axhline(90, color='gray', linestyle='--', linewidth=1.5,
                label='~solve threshold (+90)')
    plt.title('Episode reward (higher = better)')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- Success rate --- #
    plt.subplot(1, 2, 2)
    for mode, data in agg.items():
        solved = data["solved"]                   # [seeds, episodes]
        mean = solved.mean(axis=0)
        ma = moving_average(mean, ma_window)
        ma_x = np.arange(ma_window, len(mean) + 1)
        plt.plot(ma_x, ma, color=colors[mode], linewidth=2.5, label=labels[mode])
    plt.title(f'Success rate (reached goal), {ma_window}-ep moving avg')
    plt.xlabel('Episode')
    plt.ylabel('Fraction of episodes solved')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_mountaincar.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"=> Comparison graph saved to: {path}")


def main():
    models_dir, results_dir = get_save_paths()

    state_dim = 2
    action_dim = 1
    max_action = 1.0

    # --- Experiment configuration --- #
    MAX_EPISODES = 150
    BATCH_SIZE = 128
    WARMUP_STEPS = 1000
    SIGMA_INIT = 1.5        # stronger parameter noise: enough to crest the hill
    XI_MAX = 0.5
    SEEDS = [0, 1, 2]       # multi-seed for a robust claim
    MA_WINDOW = 10
    USE_SHAPING = False     # PURE SPARSE: the hard-exploration regime where
                            # NROWAN's coherent noise should beat vanilla's

    agg = {}
    for mode in ["nrowan", "vanilla"]:
        per_seed = {"rewards": [], "solved": [], "lengths": [], "sigma": []}
        for seed in SEEDS:
            print(f"\n=== Training [{mode}] seed={seed} for {MAX_EPISODES} episodes ===")
            agent, res = run_training(
                mode, seed, state_dim, action_dim, max_action,
                MAX_EPISODES, WARMUP_STEPS, BATCH_SIZE, SIGMA_INIT, XI_MAX,
                use_shaping=USE_SHAPING)
            for k in per_seed:
                per_seed[k].append(res[k])
            torch.save(agent.actor.state_dict(),
                       os.path.join(models_dir, f'actor_{mode}_seed{seed}.pth'))

        agg[mode] = {k: np.array(v, dtype=float) for k, v in per_seed.items()}
        np.savetxt(os.path.join(results_dir, f"rewards_{mode}.txt"), agg[mode]["rewards"])
        np.savetxt(os.path.join(results_dir, f"solved_{mode}.txt"), agg[mode]["solved"])
        np.savetxt(os.path.join(results_dir, f"lengths_{mode}.txt"), agg[mode]["lengths"])

    plot_comparison(agg, results_dir, MA_WINDOW)

    # --- Final verdict summary (last-30-episode averages, across seeds) --- #
    print("\n============= SUMMARY (last-30-episode averages over seeds) =============")
    print(f"{'method':18s} {'reward':>10s} {'success%':>10s} {'len':>8s}")
    for mode in ["nrowan", "vanilla"]:
        r = agg[mode]["rewards"][:, -30:].mean()
        s = agg[mode]["solved"][:, -30:].mean() * 100
        l = agg[mode]["lengths"][:, -30:].mean()
        print(f"{mode:18s} {r:10.1f} {s:10.1f} {l:8.0f}")
    print("=========================================================================")

    # --- sigma diagnostic: confirm NROWAN's output-layer noise is LEARNED
    # (rises while exploring) and then anneals, rather than collapsing at once. --- #
    sig = agg["nrowan"]["sigma"]   # [seeds, episodes]
    print("\n--- NROWAN output-layer sigma (mean over seeds) ---")
    print(f"  init={sig[:, 0].mean():.4f}  max={sig.mean(axis=0).max():.4f}  "
          f"final={sig[:, -1].mean():.4f}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
