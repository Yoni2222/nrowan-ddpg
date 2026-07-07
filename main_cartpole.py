"""
NROWAN-DQN reproduction on CartPole (one of the paper's four environments).

Compares the paper's three algorithms -- DQN, NoisyNet-DQN, NROWAN-DQN -- using
the SAME NoisyLinear + noise-reduction machinery as the continuous experiments.
This is the fast sanity check that our NROWAN implementation is correct: in the
discrete regime it was designed for, NROWAN-DQN should match or beat DQN and
show more stable learning (the paper's claims).

The paper's own description of the environment ("...the game is over when the
total reward reaches '+200'...", Sec. 5.1) matches CartPole-v0's 200-step cap,
NOT CartPole-v1's 500-step cap. We reproduce that exact cap via
gym.make(..., max_episode_steps=200) on top of CartPole-v1's physics (v0 and
v1 share identical dynamics/reward; they differ only in the step cap).

Hyperparameters match the paper's Table 1 ("Others" column) and Table 2
("Cartpole" column) exactly:
  - training budget: 30,000 environment frames (NOT a fixed episode count)
  - min frames before learning starts: 32
  - replay buffer capacity: 10,000
  - batch size: 32, target update: every 1000 steps, gamma: 0.99
  - learning rate: 0.0001, sigma0: 0.4, k_final: 4.0
  - 5 training instances (seeds) per algorithm, as in the paper's Table 3

Reports mean +/- std of the episode return across 5 seeds.
"""
import os
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from agent.dqn_agent import DQNAgent
from agent.memory import ReplayBuffer

ENV_ID = "CartPole-v1"
MAX_EPISODE_STEPS = 200          # reproduces CartPole-v0's cap (paper Sec. 5.1)
STATE_DIM, N_ACTIONS = 4, 2
INF_R, SUP_R = 0.0, 200.0         # return range under the 200-step cap (online weight k)
SOLVED_THRESHOLD = 195.0          # OpenAI Gym's official "solved" bar for CartPole-v0


def moving_average(data, window):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def run_training(mode, seed, budget_steps, lr, target_update, min_start, batch_size):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS)
    agent = DQNAgent(STATE_DIM, N_ACTIONS, mode=mode, arch="mlp",
                     lr=lr, gamma=0.99, target_update=target_update,
                     sigma_init=0.4, k_final=4.0, inf_R=INF_R, sup_R=SUP_R,
                     eps_start=1.0, eps_end=0.02, eps_decay_steps=budget_steps)
    buffer = ReplayBuffer(STATE_DIM, 1, max_size=10000)

    returns, sigmas = [], []
    pbar = tqdm(total=budget_steps, desc=f"{mode:8s} seed={seed}")
    ep = 0
    while agent.total_steps < budget_steps:
        obs, _ = env.reset(seed=seed + ep)
        state = np.asarray(obs, dtype=np.float32)
        ep_ret = 0.0
        while True:
            a = agent.select_action(state, explore=True)
            pbar.update(1)
            nobs, r, term, trunc, _ = env.step(a)
            ns = np.asarray(nobs, dtype=np.float32)
            done = term                       # bootstrap zeroed only on real terminal
            buffer.add(state, [a], r, ns, float(done))
            agent.update_k_step(r)
            if buffer.size > min_start:
                agent.train(buffer, batch_size)
            state = ns
            ep_ret += r
            if term or trunc or agent.total_steps >= budget_steps:
                break
        agent.end_episode()
        returns.append(ep_ret)
        sigmas.append(agent.noise_magnitude())
        ep += 1

    pbar.close()
    env.close()
    return {"returns": returns, "sigma": sigmas}


def plot_comparison(agg, results_dir, ma_window, n_seeds):
    """agg[mode]['returns'] is a ragged list of per-seed return lists (episode
    counts differ across seeds/modes since training is budgeted by env steps,
    not episode count). Truncate each mode to its shortest seed run before
    computing the mean +/- std band."""
    colors = {"dqn": "gray", "noisynet": "royalblue", "nrowan": "green"}
    labels = {"dqn": "DQN", "noisynet": "NoisyNet-DQN", "nrowan": "NROWAN-DQN (ours)"}

    def smoothed_mean_std(per_seed_returns):
        min_len = min(len(r) for r in per_seed_returns)
        arr = np.array([r[:min_len] for r in per_seed_returns], dtype=float)
        sm = np.array([moving_average(arr[s], ma_window) for s in range(arr.shape[0])])
        return sm.mean(axis=0), sm.std(axis=0)

    plt.figure(figsize=(9, 5.5))
    for mode in ["dqn", "noisynet", "nrowan"]:
        mean, std = smoothed_mean_std(agg[mode]["returns"])
        x = np.arange(ma_window, ma_window + len(mean))
        plt.plot(x, mean, color=colors[mode], linewidth=2.5, label=labels[mode])
        plt.fill_between(x, mean - std, mean + std, color=colors[mode], alpha=0.18)
    plt.axhline(SOLVED_THRESHOLD, color='black', linestyle='--', linewidth=1.2,
               label=f'solved ({SOLVED_THRESHOLD:.0f})')
    plt.title(f'CartPole (200-step cap) return, {ma_window}-ep MA  [mean $\\pm$ std, {n_seeds} seeds]')
    plt.xlabel('Episode'); plt.ylabel('Episode return')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_cartpole.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"=> Comparison graph saved to: {path}")


def main():
    results_dir = "results_cartpole"
    os.makedirs(results_dir, exist_ok=True)

    # --- Table 1 ("Others" column) + Table 2 ("Cartpole" column) --- #
    BUDGET_STEPS = 30_000      # training budget in env frames, not episodes
    LR = 1e-4
    TARGET_UPDATE = 1000
    MIN_START = 32
    BATCH_SIZE = 32
    SEEDS = [0, 1, 2, 3, 4]    # 5 instances, as in the paper's Table 3
    MA_WINDOW = 10

    agg = {}
    for mode in ["dqn", "noisynet", "nrowan"]:
        per_seed = {"returns": [], "sigma": []}
        for seed in SEEDS:
            print(f"\n=== Training [{mode}] seed={seed} for {BUDGET_STEPS} steps ===")
            res = run_training(mode, seed, BUDGET_STEPS, LR, TARGET_UPDATE,
                               MIN_START, BATCH_SIZE)
            for key in per_seed:
                per_seed[key].append(res[key])
        agg[mode] = per_seed         # ragged: episode count differs per seed
        for seed, r in zip(SEEDS, per_seed["returns"]):
            np.savetxt(os.path.join(results_dir, f"returns_{mode}_seed{seed}.txt"), r)

    plot_comparison(agg, results_dir, MA_WINDOW, len(SEEDS))

    # --- SUMMARY: last-50-ep mean return, mean +/- std across the 5 seeds --- #
    print(f"\n======== SUMMARY (last-50-ep mean return: mean +/- std over {len(SEEDS)} seeds) ========")
    print(f"{'method':16s} {'return':>18s}")
    for mode in ["dqn", "noisynet", "nrowan"]:
        r = np.array([np.mean(run[-50:]) for run in agg[mode]["returns"]])  # one value per seed
        print(f"{mode:16s} {r.mean():8.1f} +/- {r.std():6.1f}")
    print("=============================================================================")


if __name__ == "__main__":
    main()
