"""
NROWAN-DQN reproduction on MountainCar-v0 and Acrobot-v1 -- the paper's two
remaining classic-control environments (alongside CartPole, see
main_cartpole.py, which this script mirrors).

Faithful to the paper's Table 1 ("Others" column), Table 2, and Sec. 5.2-5.3:
  - MLP Q-net: two hidden layers of 128 (Table 1)
  - budget: 30,000 environment frames per instance (not an episode count)
  - batch 32, gamma 0.99, target update every 1000 steps, learning every step
  - min frames before learning: 32; replay capacity: 10,000
  - learning rate 0.001 for BOTH MountainCar and Acrobot (Table 2 -- note
    this differs from CartPole/Pong's 0.0001)
  - sigma0 = 0.4, k_final = 4.0
  - evaluation: after training, 64 no-exploration rounds per instance;
    5 instances per algorithm (Sec. 5.3, Table 3)
  - DQN baseline exploration: Mnih et al. (2015) taken literally -- epsilon
    1.0 -> 0.1 over 1M frames (so ~1.0 for the whole 30K budget), eval
    epsilon 0.05. The paper leaves its epsilon schedule unstated; this
    reading was validated by our CartPole reproduction.

Paper's Table 3 scores:
  MountainCar: DQN -131.90+/-21.09 | NoisyNet -128.37+/-21.97 | NROWAN -121.85+/-19.88
  Acrobot:     DQN  -87.24+/-22.33 | NoisyNet  -86.57+/-29.32 | NROWAN  -84.41+/-15.58

  python main_classic_dqn.py                    # both envs, full protocol
  python main_classic_dqn.py --env mountaincar  # one env only
"""
import argparse
import os
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from agent.dqn_agent import DQNAgent
from agent.memory import ReplayBuffer

# Per-env config. INF_R/SUP_R bound the within-episode cumulative reward for
# the online weight k (eq. 12); the paper doesn't publish its values, so we
# use each env's natural return range: worst case (time out at -1/step) up to
# a near-optimal solve.
ENVS = {
    "mountaincar": dict(
        env_id="MountainCar-v0", state_dim=2, n_actions=3,
        inf_R=-200.0, sup_R=-90.0,          # 200-step cap; fast solves ~ -90..-110
        paper="DQN -131.90+/-21.09 | NoisyNet -128.37+/-21.97 | NROWAN -121.85+/-19.88",
    ),
    "acrobot": dict(
        env_id="Acrobot-v1", state_dim=6, n_actions=3,
        inf_R=-500.0, sup_R=-60.0,          # 500-step cap; fast solves ~ -60..-100
        paper="DQN -87.24+/-22.33 | NoisyNet -86.57+/-29.32 | NROWAN -84.41+/-15.58",
    ),
}

# Table 1 ("Others") + Table 2
BUDGET_STEPS = 30_000
LR = 1e-3                        # Table 2: 0.001 for MountainCar AND Acrobot
TARGET_UPDATE = 1000
MIN_START = 32
BATCH_SIZE = 32
SEEDS = [0, 1, 2, 3, 4]          # 5 instances (Sec. 5.3)
EVAL_ROUNDS = 64
MA_WINDOW = 10

# DQN baseline exploration -- Mnih et al. (2015), see module docstring
EPS_END = 0.1
EPS_DECAY_STEPS = 1_000_000
EVAL_EPS = 0.05


def moving_average(data, window):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def run_training(cfg, mode, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(cfg["env_id"])
    agent = DQNAgent(cfg["state_dim"], cfg["n_actions"], mode=mode, arch="mlp",
                     lr=LR, gamma=0.99, target_update=TARGET_UPDATE,
                     sigma_init=0.4, k_final=4.0,
                     inf_R=cfg["inf_R"], sup_R=cfg["sup_R"],
                     eps_start=1.0, eps_end=EPS_END,
                     eps_decay_steps=EPS_DECAY_STEPS)
    buffer = ReplayBuffer(cfg["state_dim"], 1, max_size=10000)

    returns = []
    train_successes = 0              # episodes that reached the goal (term=True)
    pbar = tqdm(total=BUDGET_STEPS, desc=f"{mode:8s} seed={seed}")
    ep = 0
    while agent.total_steps < BUDGET_STEPS:
        obs, _ = env.reset(seed=seed + ep)
        state = np.asarray(obs, dtype=np.float32)
        ep_ret = 0.0
        while True:
            a = agent.select_action(state, explore=True)
            pbar.update(1)
            nobs, r, term, trunc, _ = env.step(a)
            ns = np.asarray(nobs, dtype=np.float32)
            done = term                   # bootstrap zeroed only on real terminal
            buffer.add(state, [a], r, ns, float(done))
            agent.update_k_step(r)
            if buffer.size > MIN_START:
                agent.train(buffer, BATCH_SIZE)
            state = ns
            ep_ret += r
            if term:
                train_successes += 1
            if term or trunc or agent.total_steps >= BUDGET_STEPS:
                break
        agent.end_episode()
        returns.append(ep_ret)
        ep += 1
    pbar.close()
    print(f"    [{mode} seed={seed}] training: {ep} episodes, "
          f"{train_successes} reached the goal", flush=True)

    # --- Evaluation, paper Sec. 5.3: 64 rounds, no exploration. DQN keeps
    # Mnih's eval epsilon of 0.05; noisy agents act on mean weights. --- #
    eval_returns = []
    for ev in range(EVAL_ROUNDS):
        obs, _ = env.reset(seed=10_000 + seed * EVAL_ROUNDS + ev)
        state = np.asarray(obs, dtype=np.float32)
        ep_ret = 0.0
        while True:
            if mode == "dqn" and np.random.rand() < EVAL_EPS:
                a = np.random.randint(cfg["n_actions"])
            else:
                a = agent.select_action(state, explore=False)
            nobs, r, term, trunc, _ = env.step(a)
            state = np.asarray(nobs, dtype=np.float32)
            ep_ret += r
            if term or trunc:
                break
        eval_returns.append(ep_ret)

    env.close()
    return {"returns": returns, "eval": eval_returns}


def plot_comparison(agg, results_dir, env_name):
    colors = {"dqn": "gray", "noisynet": "royalblue", "nrowan": "green"}
    labels = {"dqn": "DQN", "noisynet": "NoisyNet-DQN", "nrowan": "NROWAN-DQN (ours)"}

    def smoothed_mean_std(per_seed_returns):
        min_len = min(len(r) for r in per_seed_returns)
        arr = np.array([r[:min_len] for r in per_seed_returns], dtype=float)
        sm = np.array([moving_average(arr[s], MA_WINDOW) for s in range(arr.shape[0])])
        return sm.mean(axis=0), sm.std(axis=0)

    plt.figure(figsize=(9, 5.5))
    for mode in ["dqn", "noisynet", "nrowan"]:
        mean, std = smoothed_mean_std(agg[mode]["returns"])
        x = np.arange(MA_WINDOW, MA_WINDOW + len(mean))
        plt.plot(x, mean, color=colors[mode], linewidth=2.5, label=labels[mode])
        plt.fill_between(x, mean - std, mean + std, color=colors[mode], alpha=0.18)
    plt.title(f'{env_name} training return, {MA_WINDOW}-ep MA  '
              f'[mean $\\pm$ std, {len(SEEDS)} seeds]')
    plt.xlabel('Episode'); plt.ylabel('Episode return')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    path = os.path.join(results_dir, f'comparison_{env_name}.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"=> Comparison graph saved to: {path}")


def run_env(name, modes, seeds):
    cfg = ENVS[name]
    results_dir = f"results_{name}"
    os.makedirs(results_dir, exist_ok=True)

    agg = {}
    for mode in modes:
        per_seed = {"returns": [], "eval": []}
        for seed in seeds:
            print(f"\n=== [{name}] Training [{mode}] seed={seed} for {BUDGET_STEPS} steps ===")
            res = run_training(cfg, mode, seed)
            for key in per_seed:
                per_seed[key].append(res[key])
        agg[mode] = per_seed
        for seed, r in zip(seeds, per_seed["returns"]):
            np.savetxt(os.path.join(results_dir, f"returns_{mode}_seed{seed}.txt"), r)
        np.savetxt(os.path.join(results_dir, f"eval_{mode}.txt"),
                   np.array(per_seed["eval"], dtype=float))

    if len(modes) == 3:
        plot_comparison(agg, results_dir, name)

    print(f"\n==== [{name}] SUMMARY (Table 3 protocol: {EVAL_ROUNDS} eval rounds x "
          f"{len(seeds)} instances, budget={BUDGET_STEPS}) ====")
    print(f"paper: {cfg['paper']}")
    for mode in modes:
        ev = np.array(agg[mode]["eval"], dtype=float)
        print(f"{mode:16s} {ev.mean(axis=1).mean():8.2f} +/- {ev.std(axis=1).mean():6.2f}")
    print("=" * 77)


def main():
    global BUDGET_STEPS, EPS_DECAY_STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="both",
                    choices=["mountaincar", "acrobot", "both"])
    ap.add_argument("--budget", type=int, default=BUDGET_STEPS,
                    help="training budget in env steps (paper states 30K only "
                         "for CartPole/Pong; unstated for these two envs)")
    ap.add_argument("--modes", default="dqn,noisynet,nrowan",
                    help="comma list of algorithms to run")
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="comma list of seeds (paper: 5 instances)")
    ap.add_argument("--eps-decay-steps", type=int, default=EPS_DECAY_STEPS,
                    help="DQN epsilon anneal horizon (default: Mnih's 1M)")
    args = ap.parse_args()

    BUDGET_STEPS = args.budget
    EPS_DECAY_STEPS = args.eps_decay_steps
    modes = args.modes.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    for name in (["mountaincar", "acrobot"] if args.env == "both" else [args.env]):
        run_env(name, modes, seeds)


if __name__ == "__main__":
    main()
