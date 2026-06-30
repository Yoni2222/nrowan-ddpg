"""
NROWAN-DQN reproduction on Pong (Atari) -- the paper's flagship environment.

Same NROWAN machinery (NoisyLinear + output-layer noise-reduction loss D +
online weight k) as CartPole, but with the CNN Q-network on raw pixels. This
is the heavy reproduction: it needs a GPU and runs for hundreds of thousands of
agent steps. Run CartPole FIRST to confirm the implementation, then this.

Compares the paper's three algorithms: DQN, NoisyNet-DQN, NROWAN-DQN.

Preprocessing follows standard Atari DQN: grayscale, 84x84, frame-skip 4,
4-frame stack. Frames are stored as uint8 in the replay buffer (CNNQNet
divides by 255 internally).

  pip install "gymnasium[atari,accept-rom-license]" ale-py
"""
import os
import time
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from agent.dqn_agent import DQNAgent
from agent.memory import ImageReplayBuffer

# AtariPreprocessing + frame stacking. Wrapper names changed across gymnasium
# versions, so import defensively.
from gymnasium.wrappers import AtariPreprocessing
try:                                              # gymnasium >= 1.0
    from gymnasium.wrappers import FrameStackObservation as FrameStack
except ImportError:                               # older gymnasium
    from gymnasium.wrappers import FrameStack

ENV_ID = "ALE/Pong-v5"
N_STACK = 4
INF_R, SUP_R = -21.0, 21.0       # Pong episode-score range (for online weight k)


def make_env(seed):
    # frameskip=1 on the base env; AtariPreprocessing does the skipping/max-pool.
    env = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0)
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84,
                             grayscale_obs=True, scale_obs=False)
    env = FrameStack(env, N_STACK)               # -> obs shape (4, 84, 84) uint8
    env.action_space.seed(seed)
    return env


def moving_average(data, window):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def run_training(mode, seed, total_steps, lr, target_update, min_start,
                 train_freq, batch_size, buffer_size, log_every):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(seed)
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape       # (4, 84, 84)

    agent = DQNAgent(state_dim=None, n_actions=n_actions, mode=mode, arch="cnn",
                     lr=lr, gamma=0.99, target_update=target_update,
                     sigma_init=0.4, k_final=4.0, inf_R=INF_R, sup_R=SUP_R,
                     eps_start=1.0, eps_end=0.01, eps_decay_steps=int(total_steps * 0.5),
                     in_channels=N_STACK)
    buffer = ImageReplayBuffer(obs_shape, max_size=buffer_size)

    ep_returns, ep_steps_at = [], []              # episode score + the step it ended
    obs, _ = env.reset(seed=seed)
    state = np.asarray(obs, dtype=np.uint8)
    ep_ret = 0.0
    t0 = time.time()

    for step in range(1, total_steps + 1):
        a = agent.select_action(state, explore=True)
        nobs, r, term, trunc, _ = env.step(a)
        ns = np.asarray(nobs, dtype=np.uint8)
        done = term                               # bootstrap zeroed on real terminal only
        buffer.add(state, a, r, ns, float(done))
        agent.update_k_step(r)

        if buffer.size > min_start and step % train_freq == 0:
            agent.train(buffer, batch_size)

        state = ns
        ep_ret += r

        if term or trunc:
            agent.end_episode()
            ep_returns.append(ep_ret)
            ep_steps_at.append(step)
            ep_ret = 0.0
            obs, _ = env.reset()
            state = np.asarray(obs, dtype=np.uint8)

        if step % log_every == 0:
            recent = np.mean(ep_returns[-20:]) if ep_returns else float('nan')
            elapsed = time.time() - t0
            print(f"[{mode:8s} seed={seed}] step {step:>8d}/{total_steps}  "
                  f"avg20={recent:6.1f}  sigma={agent.noise_magnitude():.3f}  "
                  f"k={agent.k:.2f}  {elapsed/60:.1f} min")

    env.close()
    return {"returns": ep_returns, "steps_at": ep_steps_at}


def plot_comparison(results, results_dir, ma_window):
    colors = {"dqn": "gray", "noisynet": "royalblue", "nrowan": "green"}
    labels = {"dqn": "DQN", "noisynet": "NoisyNet-DQN", "nrowan": "NROWAN-DQN (ours)"}

    plt.figure(figsize=(9, 5.5))
    for mode, res in results.items():
        rets = res["returns"]
        sm = moving_average(rets, ma_window)
        x = np.arange(len(sm)) + ma_window
        plt.plot(x, sm, color=colors[mode], linewidth=2.2, label=labels[mode])
    plt.axhline(21, color='black', linestyle=':', linewidth=1.0)
    plt.axhline(-21, color='black', linestyle=':', linewidth=1.0)
    plt.title(f'Pong episode score, {ma_window}-ep moving average')
    plt.xlabel('Episode'); plt.ylabel('Episode score (win = +21)')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_pong.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"=> Comparison graph saved to: {path}")


def main():
    results_dir = "results_pong"
    os.makedirs(results_dir, exist_ok=True)

    # ---- config ---- #
    # QUICK_TEST: tiny budget just to confirm it runs end-to-end on Colab.
    # Set to False for the real reproduction (will take several GPU-hours).
    QUICK_TEST = True

    if QUICK_TEST:
        TOTAL_STEPS = 20_000
        MIN_START = 1_000
        BUFFER_SIZE = 20_000
        LOG_EVERY = 2_000
    else:
        TOTAL_STEPS = 2_000_000        # ~8M frames; Pong usually solves well before this
        MIN_START = 10_000
        BUFFER_SIZE = 100_000          # ~5.6 GB of uint8 frames (state + next_state)
        LOG_EVERY = 20_000

    LR = 1e-4
    TARGET_UPDATE = 2_500              # gradient updates between hard target syncs
    TRAIN_FREQ = 4                     # one gradient step per 4 env steps (standard Atari)
    BATCH_SIZE = 32
    SEED = 0
    MA_WINDOW = 20

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found -- Pong on CPU is impractically slow. "
              "Run this on a Colab GPU runtime.")

    results = {}
    for mode in ["dqn", "noisynet", "nrowan"]:
        print(f"\n=== Training [{mode}] for {TOTAL_STEPS} steps (seed={SEED}) ===")
        res = run_training(mode, SEED, TOTAL_STEPS, LR, TARGET_UPDATE, MIN_START,
                           TRAIN_FREQ, BATCH_SIZE, BUFFER_SIZE, LOG_EVERY)
        results[mode] = res
        np.savetxt(os.path.join(results_dir, f"returns_{mode}.txt"),
                   np.array(res["returns"], dtype=float))

    plot_comparison(results, results_dir, MA_WINDOW)

    print("\n======== SUMMARY (mean of last 20 episodes' score) ========")
    print(f"{'method':16s} {'last-20 mean score':>20s}")
    for mode in ["dqn", "noisynet", "nrowan"]:
        rets = results[mode]["returns"]
        last = np.mean(rets[-20:]) if rets else float('nan')
        print(f"{mode:16s} {last:20.1f}")
    print("===========================================================")


if __name__ == "__main__":
    main()
