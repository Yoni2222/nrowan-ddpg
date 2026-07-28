"""
NROWAN-DDPG vs Vanilla DDPG on CityLearn (building energy coordination).

CityLearn is a continuous-control benchmark for demand response: an agent
charges/discharges each building's storage (battery, DHW, cooling) to reduce
electricity cost, carbon emissions and peak demand. We run it in central-agent
mode, so the whole district is one flat continuous action vector -- a direct
drop-in for our DDPG agents.

Three agent variants (identical to the other continuous experiments):
  * vanilla     -- plain DDPG + decaying Gaussian action noise
  * nrowan      -- NoisyLinear actor, D penalty, online weight (paper transfer)
  * nrowan_iso  -- as nrowan, but the policy loss is computed through the
                   mean-only forward pass, so sigma gets no policy gradient

IMPORTANT EXPECTATION: CityLearn has a DENSE reward (cost/emissions/peak are
scored every timestep), so exploration is NOT the bottleneck here. This is the
same regime in which the grid2op experiment failed. Treat a negative result as
the likely outcome and an informative one -- it maps the boundary of where the
NROWAN mechanism applies.

Two environment-specific adaptations, both applied IDENTICALLY to all methods:
  1. Running observation standardization -- CityLearn observations mix hours,
     temperatures, kWh and prices, whose raw scales differ by orders of
     magnitude and would otherwise swamp the networks.
  2. Action rescaling from the agent's [-1, 1] (tanh) range onto each
     dimension's actual Box bounds.

  pip install CityLearn
  python main_citylearn.py --smoke                    # 2-minute sanity check
  python main_citylearn.py --noise-decay 1.0          # full 3-way comparison
  python main_citylearn.py --noise-decay 1.0 --resume # continue after a drop
"""
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from citylearn.citylearn import CityLearnEnv

from agent.ddpg_agent import DDPGAgent
from agent.memory import ReplayBuffer

DEFAULT_SCHEMA = "citylearn_challenge_2022_phase_1"
METRICS = ["rewards", "lengths", "sigma"]

COLORS = {"nrowan": "green", "nrowan_iso": "royalblue", "vanilla": "darkorange"}
LABELS = {"nrowan": "NROWAN-DDPG (original transfer)",
          "nrowan_iso": "NROWAN-DDPG-ISO (gradient-isolated, ours)",
          "vanilla": "Vanilla DDPG"}


class RunningNorm:
    """Welford running mean/std standardizer for observations. Shared shape
    across methods; each run keeps its own statistics."""

    def __init__(self, dim, clip=10.0):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def __call__(self, x, update=True):
        x = np.asarray(x, dtype=np.float64)
        if update:
            self.count += 1.0
            delta = x - self.mean
            self.mean += delta / self.count
            self.var += (delta * (x - self.mean) - self.var) / self.count
        std = np.sqrt(np.maximum(self.var, 1e-8))
        return np.clip((x - self.mean) / std, -self.clip, self.clip).astype(np.float32)


def make_env(schema, episode_steps):
    kwargs = dict(central_agent=True)
    if episode_steps:
        kwargs["episode_time_steps"] = episode_steps
    return CityLearnEnv(schema, **kwargs)


def flat_obs(obs):
    """CityLearn returns a list (central agent) or nested list; flatten it."""
    return np.asarray(obs, dtype=np.float32).ravel()


def scalar_reward(r):
    """Reward may come back as a scalar or a per-agent list."""
    return float(np.sum(np.asarray(r, dtype=np.float64)))


def env_spaces(env):
    """Action bounds for the single central-agent action vector."""
    box = env.action_space[0]
    return (np.asarray(box.low, dtype=np.float32),
            np.asarray(box.high, dtype=np.float32))


def moving_average(data, window):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def run_training(mode, seed, schema, episode_steps, n_episodes, warmup_steps,
                 batch_size, sigma_init, xi_max, noise_decay):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(schema, episode_steps)
    low, high = env_spaces(env)
    action_dim = int(low.shape[0])

    obs, _ = env.reset()
    state_raw = flat_obs(obs)
    state_dim = int(state_raw.shape[0])
    normalizer = RunningNorm(state_dim)

    # CityLearn's reward is dense and unbounded, so there is no known reward
    # range for eq. 12 -- fall back to the online min/max adjuster (as grid2op).
    agent = DDPGAgent(state_dim, action_dim, max_action=1.0,
                      sigma_init=sigma_init, xi_max=xi_max, mode=mode,
                      expl_noise_decay=noise_decay)
    buffer = ReplayBuffer(state_dim, action_dim)

    ep_rewards, ep_lengths, ep_sigma = [], [], []
    total_steps = 0

    for episode in tqdm(range(n_episodes), desc=f"{mode:11s} seed={seed}"):
        obs, _ = env.reset()
        state = normalizer(flat_obs(obs))
        ep_reward, length = 0.0, 0
        agent.reset_exploration_noise()      # coherent per-episode noise

        while True:
            if total_steps < warmup_steps:
                action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
            else:
                action = agent.select_action(state, explore=True).astype(np.float32)

            # agent acts in [-1, 1]; map onto this env's actual Box bounds
            env_action = low + (action + 1.0) * 0.5 * (high - low)
            next_obs, reward, terminated, truncated, _ = env.step([env_action])
            r = scalar_reward(reward)
            next_state = normalizer(flat_obs(next_obs))
            done = bool(terminated)          # truncation must NOT zero the bootstrap

            buffer.add(state, action, r, next_state, float(done))

            if total_steps >= warmup_steps and buffer.size > batch_size:
                agent.train(buffer, batch_size)

            state = next_state
            ep_reward += r
            length += 1
            total_steps += 1

            if terminated or truncated:
                break

        ep_rewards.append(ep_reward)
        ep_lengths.append(length)
        ep_sigma.append(agent.noise_magnitude())
        agent.update_noise_weight(ep_reward)

    env.close()
    return {"rewards": ep_rewards, "lengths": ep_lengths, "sigma": ep_sigma}


# ----------------------------- persistence ----------------------------- #

def get_results_dir():
    """Prefer Google Drive so results survive a Colab disconnect / VM reset."""
    drive = '/content/drive/MyDrive/'
    if os.path.exists(drive):
        print("Google Drive detected: results will sync to cloud.")
        results_dir = os.path.join(drive, 'NROWAN_DDPG_Project', 'results_citylearn')
    else:
        print("Google Drive not found: saving to the local project directory.")
        results_dir = "results_citylearn"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def seed_file(results_dir, mode, seed):
    return os.path.join(results_dir, f"run_{mode}_seed{seed}.npz")


def save_seed(results_dir, mode, seed, res):
    np.savez(seed_file(results_dir, mode, seed),
             **{k: np.asarray(res[k], dtype=float) for k in METRICS})


def load_seed(results_dir, mode, seed):
    path = seed_file(results_dir, mode, seed)
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in METRICS}


def plot_comparison(agg, results_dir, ma_window, n_seeds):
    def smoothed_mean_std(arr):
        sm = np.array([moving_average(arr[s], ma_window) for s in range(arr.shape[0])])
        return sm.mean(axis=0), sm.std(axis=0)

    plt.figure(figsize=(13, 5))

    plt.subplot(1, 2, 1)
    for mode, data in agg.items():
        mean, std = smoothed_mean_std(data["rewards"])
        x = np.arange(ma_window, ma_window + len(mean))
        plt.plot(x, mean, color=COLORS[mode], linewidth=2.5, label=LABELS[mode])
        plt.fill_between(x, mean - std, mean + std, color=COLORS[mode], alpha=0.20)
    plt.title(f'CityLearn episode reward (higher = better)  '
              f'[mean $\\pm$ std, {n_seeds} seeds]')
    plt.xlabel('Episode'); plt.ylabel('Episode reward')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    for mode, data in agg.items():
        if mode == "vanilla":
            continue
        mean = data["sigma"].mean(axis=0)
        plt.plot(np.arange(1, len(mean) + 1), mean, color=COLORS[mode],
                 linewidth=2.2, label=LABELS[mode])
    plt.title('Output-layer sigma (exploration noise actually retained)')
    plt.xlabel('Episode'); plt.ylabel('mean |sigma|')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_citylearn.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"=> Comparison graph saved to: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default=DEFAULT_SCHEMA,
                    help="CityLearn dataset/schema name")
    ap.add_argument("--modes", default="nrowan,nrowan_iso,vanilla",
                    help="comma list of agent variants to train")
    ap.add_argument("--seeds", default="0,1,2", help="comma list of seeds")
    ap.add_argument("--episodes", type=int, default=60,
                    help="training episodes per seed")
    ap.add_argument("--episode-steps", type=int, default=1000,
                    help="timesteps per episode (0 = the dataset's full length, "
                         "typically 8760 = one year, which is far slower)")
    ap.add_argument("--noise-decay", type=float, default=0.99,
                    help="per-episode decay of vanilla's Gaussian action noise. "
                         "1.0 = no decay (fair baseline). Ignored by nrowan modes")
    ap.add_argument("--resume", action="store_true",
                    help="skip and reuse any (mode, seed) already checkpointed")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (1 seed, 2 episodes, 200 steps) to verify the "
                         "install and the env wiring before committing GPU hours")
    args = ap.parse_args()

    modes = args.modes.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    n_episodes, episode_steps = args.episodes, args.episode_steps
    if args.smoke:
        modes, seeds, n_episodes, episode_steps = modes, [0], 2, 200
        print("SMOKE MODE: 1 seed, 2 episodes, 200 steps per episode.")

    results_dir = get_results_dir()

    BATCH_SIZE = 128
    WARMUP_STEPS = 500 if args.smoke else 2000
    SIGMA_INIT = 0.5
    XI_MAX = 0.5
    MA_WINDOW = 1 if args.smoke else 5

    agg = {}
    for mode in modes:
        per_seed = {k: [] for k in METRICS}
        for seed in seeds:
            cached = load_seed(results_dir, mode, seed) if args.resume else None
            if cached is not None:
                print(f"--- [{mode}] seed={seed}: reusing saved checkpoint ---")
                res = cached
            else:
                print(f"\n=== Training [{mode}] seed={seed} for {n_episodes} episodes ===")
                res = run_training(mode, seed, args.schema, episode_steps,
                                   n_episodes, WARMUP_STEPS, BATCH_SIZE,
                                   SIGMA_INIT, XI_MAX, args.noise_decay)
                if not args.smoke:
                    save_seed(results_dir, mode, seed, res)
            for k in per_seed:
                per_seed[k].append(res[k])
        agg[mode] = {k: np.array(v, dtype=float) for k, v in per_seed.items()}
        if not args.smoke:
            for k in METRICS:
                np.savetxt(os.path.join(results_dir, f"{k}_{mode}.txt"), agg[mode][k])

    if not args.smoke:
        plot_comparison(agg, results_dir, MA_WINDOW, len(seeds))

    tail = min(10, n_episodes)
    print(f"\n===== SUMMARY (last-{tail}-ep reward: mean +/- std over "
          f"{len(seeds)} seeds) =====")
    print(f"{'method':14s} {'reward':>22s}")
    for mode in modes:
        r = agg[mode]["rewards"][:, -tail:].mean(axis=1)     # one value per seed
        print(f"{mode:14s} {r.mean():12.2f} +/- {r.std():8.2f}")
    print("=" * 60)

    for mode in modes:
        if mode == "vanilla":
            continue
        sig = agg[mode]["sigma"]
        print(f"\n--- [{mode}] output-layer sigma (mean over seeds) ---")
        print(f"  init={sig[:, 0].mean():.4f}  max={sig.mean(axis=0).max():.4f}  "
              f"final={sig[:, -1].mean():.4f}")


if __name__ == "__main__":
    main()
