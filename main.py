import os
import grid2op
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from env_setup.state_extractor import extract_state
from agent.ddpg_agent import DDPGAgent
from agent.memory import ReplayBuffer


def get_save_paths():
    colab_drive_path = '/content/drive/MyDrive/'
    if os.path.exists(colab_drive_path):
        print("Google Drive detection: SUCCESS. Training data will sync to cloud.")
        base_dir = os.path.join(colab_drive_path, 'NROWAN_DDPG_Project')
        models_dir = os.path.join(base_dir, 'saved_models')
        results_dir = os.path.join(base_dir, 'results')
    else:
        print("Google Drive detection: NOT FOUND. Saving to local project directory...")
        models_dir = "saved_models"
        results_dir = "results"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return models_dir, results_dir


def moving_average(data, window):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def compute_donothing_reference(env, n_ep, max_steps):
    """Reference baseline: take the empty action every step. Returns mean episode
    length (survival) and mean violations per episode."""
    lengths, viols = [], []
    for e in range(n_ep):
        env.set_id(e % 100)
        obs = env.reset()
        L, v = 0, 0
        for _ in range(max_steps):
            obs, r, done, info = env.step(env.action_space({}))
            L += 1
            if np.max(np.nan_to_num(obs.rho, nan=0.0)) >= 1.0:
                v += 1
            if done:
                break
        lengths.append(L)
        viols.append(v)
    return float(np.mean(lengths)), float(np.mean(viols))


def run_training(env, mode, seed, redisp_mask, ramp_up, state_dim, action_dim,
                 max_action, n_episodes, max_steps, warmup_steps, batch_size,
                 sigma_init, xi_max):
    """Train one agent (mode='nrowan' or 'vanilla') and return per-episode metrics.
    The same seed is used for both methods so they face the SAME chronics order
    -> a fair, paired comparison where the only difference is the algorithm."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        env.seed(seed)
    except Exception:
        pass

    agent = DDPGAgent(state_dim, action_dim, max_action,
                      sigma_init=sigma_init, xi_max=xi_max, mode=mode)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    ep_lengths, ep_violations, ep_rewards = [], [], []
    total_steps = 0

    for episode in tqdm(range(n_episodes), desc=f"{mode:7s}"):
        env.set_id(int(np.random.randint(0, 1000)))
        obs = env.reset()
        state = extract_state(obs)

        ep_reward, ep_violation, length = 0.0, 0, 0
        agent.reset_exploration_noise()   # NROWAN: coherent per-episode noise

        for step in range(max_steps):
            if total_steps < warmup_steps:
                flat_action = np.random.uniform(
                    -max_action, max_action, size=action_dim).astype(np.float32)
            else:
                flat_action = agent.select_action(state, explore=True)

            full_redispatch = np.zeros(env.n_gen, dtype=np.float32)
            full_redispatch[redisp_mask] = flat_action * ramp_up[redisp_mask]
            next_obs, reward, done, info = env.step(
                env.action_space({"redispatch": full_redispatch}))
            next_state = extract_state(next_obs)

            max_rho = np.max(np.nan_to_num(next_obs.rho, nan=0.0))
            if max_rho >= 1.0:
                ep_violation += 1
            if max_rho > 0.8:
                reward -= (max_rho - 0.8) * 10.0
            reward = np.clip(reward, -10.0, 2.0)

            replay_buffer.add(state, flat_action, reward, next_state, done)

            if total_steps >= warmup_steps and replay_buffer.size > batch_size:
                agent.train(replay_buffer, batch_size)

            state = next_state
            ep_reward += reward
            length += 1
            total_steps += 1
            if done:
                break

        ep_lengths.append(length)
        ep_violations.append(ep_violation)
        ep_rewards.append(ep_reward)
        agent.update_noise_weight(ep_reward)

    return agent, {"lengths": ep_lengths, "violations": ep_violations, "rewards": ep_rewards}


def plot_comparison(agg, donothing, results_dir, ma_window):
    """agg[mode][key] is a 2D array [n_seeds, n_episodes]. Plots mean +/- std band."""
    dn_len, dn_viol = donothing
    colors = {"nrowan": "green", "vanilla": "darkorange"}
    labels = {"nrowan": "NROWAN-DDPG (ours)", "vanilla": "Vanilla DDPG"}

    def smoothed_mean_std(arr):
        # smooth each seed, then take mean/std across seeds
        sm = np.array([moving_average(arr[s], ma_window) for s in range(arr.shape[0])])
        return sm.mean(axis=0), sm.std(axis=0)

    plt.figure(figsize=(14, 5))

    # --- Survival (episode length) --- #
    plt.subplot(1, 2, 1)
    for mode, data in agg.items():
        mean, std = smoothed_mean_std(data["lengths"])
        x = np.arange(ma_window, ma_window + len(mean))
        plt.plot(x, mean, color=colors[mode], linewidth=2.5, label=labels[mode])
        plt.fill_between(x, mean - std, mean + std, color=colors[mode], alpha=0.20)
    plt.axhline(dn_len, color='gray', linestyle='--', linewidth=2,
                label=f'do-nothing ({dn_len:.0f})')
    plt.title('Survival: steps before blackout (higher = better)  [mean $\\pm$ std, 3 seeds]')
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- Violations --- #
    plt.subplot(1, 2, 2)
    for mode, data in agg.items():
        mean, std = smoothed_mean_std(data["violations"])
        x = np.arange(ma_window, ma_window + len(mean))
        plt.plot(x, mean, color=colors[mode], linewidth=2.5, label=labels[mode])
        plt.fill_between(x, mean - std, mean + std, color=colors[mode], alpha=0.20)
    plt.axhline(dn_viol, color='gray', linestyle='--', linewidth=2,
                label=f'do-nothing ({dn_viol:.1f})')
    plt.title('Safety violations per episode (lower = better)  [mean $\\pm$ std]')
    plt.xlabel('Episode')
    plt.ylabel('Violations (rho >= 1.0)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_nrowan_vs_vanilla.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"=> Comparison graph saved to: {path}")


def main():
    models_dir, results_dir = get_save_paths()

    print("Initializing Grid2Op environment...")
    env = grid2op.make("rte_case14_realistic")

    dummy_obs = env.reset()
    state_dim = extract_state(dummy_obs).shape[0]
    redisp_mask = np.asarray(env.gen_redispatchable, dtype=bool)
    ramp_up = np.asarray(env.gen_max_ramp_up, dtype=np.float32)
    action_dim = int(np.sum(redisp_mask))
    max_action = 1.0
    print(f"Controllable (redispatchable) generators: {action_dim} / {env.n_gen}")

    # --- Experiment configuration --- #
    MAX_EPISODES = 150
    MAX_STEPS = 2000          # long horizon: let the grid actually get stressed
    BATCH_SIZE = 128
    WARMUP_STEPS = 5000
    SIGMA_INIT = 0.5
    XI_MAX = 0.5
    SEEDS = [0, 1, 2]         # multi-seed for a robust claim (mean +/- std)
    MA_WINDOW = 20

    print("\nComputing do-nothing reference baseline...")
    donothing = compute_donothing_reference(env, n_ep=10, max_steps=MAX_STEPS)
    print(f"   do-nothing: mean length={donothing[0]:.1f} | mean violations={donothing[1]:.2f}")

    agg = {}
    for mode in ["nrowan", "vanilla"]:
        per_seed = {"lengths": [], "violations": [], "rewards": []}
        for seed in SEEDS:
            print(f"\n=== Training [{mode}] seed={seed} for {MAX_EPISODES} episodes ===")
            agent, res = run_training(
                env, mode, seed, redisp_mask, ramp_up, state_dim, action_dim,
                max_action, MAX_EPISODES, MAX_STEPS, WARMUP_STEPS, BATCH_SIZE,
                SIGMA_INIT, XI_MAX)
            for k in per_seed:
                per_seed[k].append(res[k])
            torch.save(agent.actor.state_dict(),
                       os.path.join(models_dir, f'actor_{mode}_seed{seed}.pth'))

        agg[mode] = {k: np.array(v, dtype=float) for k, v in per_seed.items()}
        for k in ["lengths", "violations", "rewards"]:
            np.savetxt(os.path.join(results_dir, f"{k}_{mode}.txt"), agg[mode][k])

    np.savetxt(os.path.join(results_dir, "donothing_reference.txt"), np.array(donothing))
    plot_comparison(agg, donothing, results_dir, MA_WINDOW)

    # --- Final verdict: mean +/- std across the 3 seeds (last-30-ep averages) --- #
    print("\n========== SUMMARY (last-30-ep averages: mean +/- std over 3 seeds) ==========")
    print(f"{'method':14s} {'survival':>20s} {'violations':>20s}")
    print(f"{'do-nothing':14s} {donothing[0]:13.1f}{'':7s} {donothing[1]:13.2f}")
    for mode in ["nrowan", "vanilla"]:
        l = agg[mode]["lengths"][:, -30:].mean(axis=1)       # one value per seed
        v = agg[mode]["violations"][:, -30:].mean(axis=1)
        print(f"{mode:14s} {l.mean():8.1f} +/- {l.std():6.1f}  {v.mean():8.2f} +/- {v.std():6.2f}")
    print("=============================================================================")


if __name__ == "__main__":
    main()
