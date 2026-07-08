"""
NROWAN-DQN reproduction on Pong (Atari) -- the paper's high-dimensional env.

Faithful to the paper's Table 1 ("Pong" column), Table 2, and Sec. 5.2-5.3:
  - grayscale, 84x84, frame stack = 1 (paper: "we don't stack frames"),
    action repetitions = 1, reward clipping = False
  - Q-net: conv 32/64/64 (8x8/4x4/3x3, stride 4/2/1) + two 512 hidden FC
  - budget: 1,000,000 environment frames per instance
  - learning: every 1 step, batch 32, gamma 0.99, target update every 1000
  - min frames before learning: 10,000; replay capacity: 100,000
  - lr = 0.0001, sigma0 = 0.4, k_final = 4.0 (Table 2)
  - evaluation: after training, 64 no-exploration rounds per instance;
    5 instances per algorithm (Sec. 5.3, Table 3)
  - DQN baseline exploration: Mnih et al. (2015) -- epsilon 1.0 -> 0.1 over
    1M frames, eval epsilon 0.05 (the paper leaves this unstated; this is the
    reading validated by our CartPole reproduction)

Paper's Table 3 Pong scores: DQN 17.07+/-3.36 | NoisyNet 17.95+/-3.08 |
NROWAN 18.81+/-2.87.

A full reproduction is 3 algorithms x 5 seeds x 1M frames -- far more than
one Colab session. Use --modes/--seeds to split the work across sessions;
each finished run is saved to results_pong/ and --summary aggregates
whatever runs exist so far.

  pip install "gymnasium[atari,accept-rom-license]" ale-py
  python main_pong.py --quick                     # end-to-end sanity check
  python main_pong.py --modes nrowan --seeds 0    # one real run
  python main_pong.py --summary                   # aggregate finished runs
"""
import argparse
import os
import time
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from agent.dqn_agent import DQNAgent
from agent.memory import ImageReplayBuffer

from gymnasium.wrappers import AtariPreprocessing

ENV_ID = "ALE/Pong-v5"
INF_R, SUP_R = -21.0, 21.0       # Pong episode-score range (for online weight k)
EVAL_ROUNDS = 64                 # post-training evaluation rounds (Sec. 5.3)
EPS_END = 0.1                    # Mnih et al. (2015) -- see module docstring
EPS_DECAY_STEPS = 1_000_000
EVAL_EPS = 0.05
RESULTS_DIR = "results_pong"

PAPER_SCORES = "DQN 17.07+/-3.36 | NoisyNet 17.95+/-3.08 | NROWAN 18.81+/-2.87"


def make_env(seed):
    # Table 1: grayscale, 84x84 down-sampling, frame stack 1, action
    # repetitions 1 -> every emulator frame is one agent step. v5's sticky
    # actions are disabled (the paper's Table 1 has no action repetition).
    env = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0)
    env = AtariPreprocessing(env, frame_skip=1, screen_size=84,
                             grayscale_obs=True, scale_obs=False,
                             noop_max=0)
    env.action_space.seed(seed)
    return env                    # obs: (84, 84) uint8


def obs_to_state(obs):
    return np.asarray(obs, dtype=np.uint8)[None, :, :]     # (1, 84, 84)


def moving_average(data, window):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def run_training(mode, seed, total_steps, lr, target_update, min_start,
                 batch_size, buffer_size, log_every):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(seed)
    n_actions = env.action_space.n

    agent = DQNAgent(state_dim=None, n_actions=n_actions, mode=mode, arch="cnn",
                     lr=lr, gamma=0.99, target_update=target_update,
                     sigma_init=0.4, k_final=4.0, inf_R=INF_R, sup_R=SUP_R,
                     eps_start=1.0, eps_end=EPS_END,
                     eps_decay_steps=EPS_DECAY_STEPS, in_channels=1)
    buffer = ImageReplayBuffer((1, 84, 84), max_size=buffer_size)

    ep_returns = []
    obs, _ = env.reset(seed=seed)
    state = obs_to_state(obs)
    ep_ret = 0.0
    t0 = time.time()

    for step in range(1, total_steps + 1):
        a = agent.select_action(state, explore=True)
        nobs, r, term, trunc, _ = env.step(a)
        ns = obs_to_state(nobs)
        done = term                               # bootstrap zeroed on real terminal only
        buffer.add(state, a, r, ns, float(done))
        agent.update_k_step(r)

        if buffer.size > min_start:               # learning every 1 step (Table 1)
            agent.train(buffer, batch_size)

        state = ns
        ep_ret += r

        if term or trunc:
            agent.end_episode()
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
            state = obs_to_state(obs)

        if step % log_every == 0:
            recent = np.mean(ep_returns[-10:]) if ep_returns else float('nan')
            elapsed = time.time() - t0
            print(f"[{mode:8s} seed={seed}] step {step:>8d}/{total_steps}  "
                  f"avg10={recent:6.1f}  sigma={agent.noise_magnitude():.3f}  "
                  f"k={agent.k:.2f}  {elapsed/60:.1f} min", flush=True)

    # --- Evaluation, paper Sec. 5.3: 64 rounds, no exploration. DQN keeps
    # Mnih's eval epsilon of 0.05; noisy agents act on mean weights. --- #
    eval_returns = []
    for ev in range(EVAL_ROUNDS):
        obs, _ = env.reset(seed=10_000 + seed * EVAL_ROUNDS + ev)
        state = obs_to_state(obs)
        ep_ret = 0.0
        while True:
            if mode == "dqn" and np.random.rand() < EVAL_EPS:
                a = np.random.randint(n_actions)
            else:
                a = agent.select_action(state, explore=False)
            nobs, r, term, trunc, _ = env.step(a)
            state = obs_to_state(nobs)
            ep_ret += r
            if term or trunc:
                break
        eval_returns.append(ep_ret)
        print(f"[{mode:8s} seed={seed}] eval round {ev+1}/{EVAL_ROUNDS}: {ep_ret:+.0f}",
              flush=True)

    env.close()
    return {"returns": ep_returns, "eval": eval_returns}


def save_run(mode, seed, res):
    np.savetxt(os.path.join(RESULTS_DIR, f"returns_{mode}_seed{seed}.txt"),
               np.array(res["returns"], dtype=float))
    np.savetxt(os.path.join(RESULTS_DIR, f"eval_{mode}_seed{seed}.txt"),
               np.array(res["eval"], dtype=float))


def summarize():
    """Aggregate whatever finished runs exist in results_pong/ using the
    paper's Table 3 protocol: instance score = mean of its 64 eval rounds,
    final score = mean over instances, std = per-instance std averaged."""
    print(f"\n==== SUMMARY (Table 3 protocol)  [paper: {PAPER_SCORES}] ====")
    print(f"{'method':16s} {'score':>16s} {'instances':>12s}")
    colors = {"dqn": "gray", "noisynet": "royalblue", "nrowan": "green"}
    labels = {"dqn": "DQN", "noisynet": "NoisyNet-DQN", "nrowan": "NROWAN-DQN (ours)"}
    plt.figure(figsize=(9, 5.5))
    for mode in ["dqn", "noisynet", "nrowan"]:
        evals, curves = [], []
        for seed in range(10):
            path = os.path.join(RESULTS_DIR, f"eval_{mode}_seed{seed}.txt")
            if os.path.exists(path):
                evals.append(np.loadtxt(path))
                curves.append(np.loadtxt(
                    os.path.join(RESULTS_DIR, f"returns_{mode}_seed{seed}.txt")))
        if not evals:
            print(f"{mode:16s} {'-- no runs --':>16s}")
            continue
        ev = np.array(evals)                                # [n_inst, 64]
        print(f"{mode:16s} {ev.mean(axis=1).mean():8.2f} +/- {ev.std(axis=1).mean():5.2f}"
              f" {len(evals):12d}")
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        sm = np.array([moving_average(a, 10) for a in arr])
        mean, std = sm.mean(axis=0), sm.std(axis=0)
        x = np.arange(10, 10 + len(mean))
        plt.plot(x, mean, color=colors[mode], linewidth=2.2, label=labels[mode])
        plt.fill_between(x, mean - std, mean + std, color=colors[mode], alpha=0.18)
    print("=" * 77)
    plt.title('Pong episode score during training, 10-ep MA')
    plt.xlabel('Episode'); plt.ylabel('Episode score (win = +21)')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'comparison_pong.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"=> Comparison graph saved to: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="dqn,noisynet,nrowan",
                    help="comma list of algorithms to run")
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="comma list of seeds (paper: 5 instances)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny 20K-step run just to verify everything works")
    ap.add_argument("--summary", action="store_true",
                    help="only aggregate existing runs, train nothing")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.summary:
        summarize()
        return

    # --- Table 1, "Pong" column --- #
    if args.quick:
        TOTAL_STEPS, MIN_START, BUFFER_SIZE, LOG_EVERY = 20_000, 1_000, 20_000, 2_000
    else:
        TOTAL_STEPS, MIN_START, BUFFER_SIZE, LOG_EVERY = 1_000_000, 10_000, 100_000, 10_000
    LR = 1e-4                     # Table 2, Pong
    TARGET_UPDATE = 1000          # Table 1: every 1000 steps

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found -- Pong on CPU is impractically slow. "
              "Run this on a Colab GPU runtime.")

    for mode in args.modes.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            print(f"\n=== Training [{mode}] seed={seed} for {TOTAL_STEPS} steps ===")
            res = run_training(mode, seed, TOTAL_STEPS, LR, TARGET_UPDATE,
                               MIN_START, batch_size=32, buffer_size=BUFFER_SIZE,
                               log_every=LOG_EVERY)
            save_run(mode, seed, res)
            ev = np.array(res["eval"])
            print(f"--- [{mode} seed={seed}] eval: {ev.mean():.2f} +/- {ev.std():.2f} "
                  f"(saved to {RESULTS_DIR}/) ---")

    summarize()


if __name__ == "__main__":
    main()
