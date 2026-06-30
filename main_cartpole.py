"""
NROWAN-DQN reproduction on CartPole-v1 (one of the paper's four environments).

Compares the paper's three algorithms -- DQN, NoisyNet-DQN, NROWAN-DQN -- using
the SAME NoisyLinear + noise-reduction machinery as the continuous experiments.
This is the fast sanity check that our NROWAN implementation is correct: in the
discrete regime it was designed for, NROWAN-DQN should match or beat DQN and
show more stable learning (the paper's claims).

Reports mean +/- std of the episode return across 3 seeds.
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
STATE_DIM, N_ACTIONS = 4, 2
INF_R, SUP_R = 0.0, 500.0        # CartPole-v1 return range (for online weight k)


def moving_average(data, window):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def run_training(mode, seed, n_episodes, lr, target_update, min_start, batch_size):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(ENV_ID)
    agent = DQNAgent(STATE_DIM, N_ACTIONS, mode=mode, arch="mlp",
                     lr=lr, gamma=0.99, target_update=target_update,
                     sigma_init=0.4, k_final=4.0, inf_R=INF_R, sup_R=SUP_R,
                     eps_start=1.0, eps_end=0.02, eps_decay_steps=5000)
    buffer = ReplayBuffer(STATE_DIM, 1, max_size=10000)

    returns, sigmas = [], []
    for ep in tqdm(range(n_episodes), desc=f"{mode:8s} seed={seed}"):
        obs, _ = env.reset(seed=seed + ep)
        state = np.asarray(obs, dtype=np.float32)
        ep_ret = 0.0
        while True:
            a = agent.select_action(state, explore=True)
            nobs, r, term, trunc, _ = env.step(a)
            ns = np.asarray(nobs, dtype=np.float32)
            done = term                       # bootstrap zeroed only on real terminal
            buffer.add(state, [a], r, ns, float(done))
            agent.update_k_step(r)
            if buffer.size > min_start:
                agent.train(buffer, batch_size)
            state = ns
            ep_ret += r
            if term or trunc:
                break
        agent.end_episode()
        returns.append(ep_ret)
        sigmas.append(agent.noise_magnitude())

    env.close()
    return {"returns": returns, "sigma": sigmas}


def plot_comparison(agg, results_dir, ma_window):
    colors = {"dqn": "gray", "noisynet": "royalblue", "nrowan": "green"}
    labels = {"dqn": "DQN", "noisynet": "NoisyNet-DQN", "nrowan": "NROWAN-DQN (ours)"}

    def smoothed_mean_std(arr):
        sm = np.array([moving_average(arr[s], ma_window) for s in range(arr.shape[0])])
        return sm.mean(axis=0), sm.std(axis=0)

    plt.figure(figsize=(9, 5.5))
    for mode in ["dqn", "noisynet", "nrowan"]:
        mean, std = smoothed_mean_std(agg[mode]["returns"])
        x = np.arange(ma_window, ma_window + len(mean))
        plt.plot(x, mean, color=colors[mode], linewidth=2.5, label=labels[mode])
        plt.fill_between(x, mean - std, mean + std, color=colors[mode], alpha=0.18)
    plt.axhline(475, color='black', linestyle='--', linewidth=1.2, label='solved (475)')
    plt.title(f'CartPole-v1 return, {ma_window}-ep MA  [mean $\\pm$ std, 3 seeds]')
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

    N_EPISODES = 700
    LR = 1e-4
    TARGET_UPDATE = 1000
    MIN_START = 200
    BATCH_SIZE = 32
    SEEDS = [0, 1, 2]
    MA_WINDOW = 10

    agg = {}
    for mode in ["dqn", "noisynet", "nrowan"]:
        per_seed = {"returns": [], "sigma": []}
        for seed in SEEDS:
            print(f"\n=== Training [{mode}] seed={seed} for {N_EPISODES} episodes ===")
            res = run_training(mode, seed, N_EPISODES, LR, TARGET_UPDATE,
                               MIN_START, BATCH_SIZE)
            for key in per_seed:
                per_seed[key].append(res[key])
        agg[mode] = {k: np.array(v, dtype=float) for k, v in per_seed.items()}
        np.savetxt(os.path.join(results_dir, f"returns_{mode}.txt"), agg[mode]["returns"])

    plot_comparison(agg, results_dir, MA_WINDOW)

    # --- SUMMARY: last-50-ep mean return, mean +/- std across the 3 seeds --- #
    print("\n======== SUMMARY (last-50-ep mean return: mean +/- std over 3 seeds) ========")
    print(f"{'method':16s} {'return':>18s}")
    for mode in ["dqn", "noisynet", "nrowan"]:
        r = agg[mode]["returns"][:, -50:].mean(axis=1)      # one value per seed
        print(f"{mode:16s} {r.mean():8.1f} +/- {r.std():6.1f}")
    print("=============================================================================")


if __name__ == "__main__":
    main()
