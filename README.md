# NROWAN - from discrete reproduction to a continuous-control autopsy

Course project (Reinforcement Learning, 2026). Two things live in this repo:

1. **reproduction** of *NROWAN-DQN: A Stable Noisy Network with Noise Reduction
   and Online Weight Adjustment for Exploration* ([arXiv:2006.10980](https://arxiv.org/abs/2006.10980))
   on the paper's own discrete environments.
2. **An attempt to port that mechanism to continuous control (DDPG)**

---

## What the mechanism is

Every weight in a noisy layer is `w = mu + sigma * eps`, where `mu` and `sigma` are learned
and `eps` is resampled noise. `sigma` is the exploration knob. The actor's loss is:

```
actor_loss  =  policy_loss  +  k * D
                    |              |
                    |              +-- D = sum of |sigma| over the OUTPUT layer (paper eq. 8)
                    +----------------- maximize the critic's value of the emitted action
```

`k` is an online weight that rises with the agent's recent performance: explore early,
settle late (paper eq. 12).

**The finding.** Both terms can change `sigma`, and they reach it by different routes:

| Path | Route | Exists in discrete control? |
|---|---|---|
| **A** - via `policy_loss` | `sigma` → weight → action → critic value → loss | **No.** The action is chosen by `argmax`, which is a comparison, not a differentiable computation. |
| **B** - via `D` | `sigma` → loss | Yes. `D` reads `sigma` directly. |

In continuous control the network *emits* the action, so path A opens and Q-maximization
suppresses the exploration mechanism as a side effect. Path B turns out to be even more
destructive: with Adam, `D`'s tiny-but-perfectly-consistent gradient still moves `sigma` a
full learning rate per step, zeroing it in ~440 updates regardless of how small `k` is.

Both paths have a switch in this codebase - see [Agent variants](#agent-variants) and
[Sigma controls](#sigma-controls).

---

## Setup

```bash
pip install -r requirements.txt
```

**CityLearn is deliberately not in `requirements.txt`.** It pins `scikit-learn<=1.2.2`,
`numpy<2.0.0` and `gymnasium<=0.28.1`; the scikit-learn pin has no wheels for modern Python
so pip tries to build it from source and fails, and the gymnasium/numpy pins would
*downgrade* the stack every other experiment depends on. Install it without its dependency
chain instead:

```bash
pip install --no-deps CityLearn
pip install simplejson pyyaml platformdirs scikit-learn
python -c "from citylearn.citylearn import CityLearnEnv; print('ok')"
```

Verify nothing was downgraded - `gymnasium` must still report 1.x:

```bash
python -c "import gymnasium; print(gymnasium.__version__)"
```

Atari (for Pong) is a large download; install only when needed:

```bash
pip install "gymnasium[atari,accept-rom-license]" ale-py
```

---

## Repository layout

```
agent/
  networks.py      NoisyLinear (dual noise buffers), Actor, Critic
  q_networks.py    MLPQNet (classic control) and CNNQNet (Atari)
  dqn_agent.py     DQNAgent  -- discrete: dqn / noisynet / nrowan
  ddpg_agent.py    DDPGAgent -- continuous: vanilla / nrowan / nrowan_iso
  noise.py         OnlineWeightAdjuster (the k / xi schedule)
  memory.py        ReplayBuffer, ImageReplayBuffer (uint8 frames)
env_setup/
  state_extractor.py   grid2op observation -> normalized state vector

main_cartpole.py     CartPole            (discrete, reproduction)
main_classic_dqn.py  MountainCar-v0 / Acrobot-v1  (discrete, reproduction)
main.py              grid2op             (continuous)
main_mountaincar.py  MountainCarContinuous-v0     (continuous)
main_pointmaze.py    PointMaze_Medium-v3 (continuous)
main_citylearn.py    CityLearn           (continuous)
```

---

## Agent variants

### Discrete (`main_cartpole.py`, `main_classic_dqn.py`) - `--modes`

| Mode | What it is |
|---|---|
| `dqn` | Plain DQN, epsilon-greedy, no noisy layers |
| `noisynet` | Noisy layers, **no** noise-reduction penalty (`k = 0`) |
| `nrowan` | Noisy layers + penalty `D` + online weight `k` - the paper's method |

### Continuous (`main.py`, `main_mountaincar.py`, `main_pointmaze.py`, `main_citylearn.py`) - `--modes`

| Mode | What it is |
|---|---|
| `vanilla` | Plain DDPG + decaying Gaussian action noise (the baseline) |
| `nrowan` | Direct port of the paper. Policy loss computed through the **noisy** forward pass, so `sigma` receives gradients from **both** paths |
| `nrowan_iso` | **Ours.** Identical, except the policy loss goes through the **mean-only** forward pass (`actor.eval()`), so `sigma` is invisible to the policy gradient. Path A closed |

Verified rather than assumed: with the penalty weight zeroed, the gradient norm reaching the
output-layer `sigma` from the policy term alone is **0.234** under `nrowan` and **exactly
0.000** under `nrowan_iso`.

---

## Sigma controls

Two independent knobs, both defaulting to the original behaviour so existing results stay
reproducible. Available on all three continuous scripts.

| Flag | Default | Effect |
|---|---|---|
| `--sigma-optimizer adam` | `adam` | Original. Adam normalizes each gradient by its own magnitude, so `D` zeroes `sigma` in ~440 updates whatever `k` is |
| `--sigma-optimizer sgd` | | **The fix.** `sigma` parameters get a plain non-normalizing optimizer while `mu` keeps Adam, restoring decay proportional to `k`. Measured over 600 updates at full penalty: `0.04419 → 0.04408` (vs `→ 0.00001` under Adam) |
| `--sigma-floor F` | `0.0` (off) | Clamp every `sigma` to at least `F` after each update. Also stops `sigma` crossing to negative values, from which `|sigma|` would push it further away |

Runs using non-default sigma settings **tag their output files** (`run_nrowan_iso_sigsgd_seed0.npz`)
so they never overwrite or get confused with default-config results.

---

## Running the experiments

### 1. Discrete reproduction - CartPole

No flags. Hyperparameters are fixed to the paper's Table 1 ("Others" column) and Table 2:
30,000 environment-step budget (not an episode count), `min_start=32`, buffer 10,000,
batch 32, target sync every 1000 steps, `lr=1e-4`, `sigma0=0.4`, `k_final=4.0`, 5 seeds,
64 post-training evaluation rounds per seed.

```bash
python main_cartpole.py
```

Runtime: minutes on CPU. Results in `results_cartpole/`.

> The paper's CartPole is the **200-step-cap** version - Sec. 5.1 says the episode ends at
> total reward "+200" - reproduced via `gym.make("CartPole-v1", max_episode_steps=200)`.

### 2. Discrete reproduction - MountainCar-v0 and Acrobot-v1

```bash
python main_classic_dqn.py                                    # both envs, full protocol
python main_classic_dqn.py --env acrobot --budget 200000      # one env, larger budget
python main_classic_dqn.py --env acrobot --modes nrowan --seeds 0
```

| Flag | Default | Notes |
|---|---|---|
| `--env` | `both` | `mountaincar` \| `acrobot` \| `both` |
| `--budget` | `30000` | Environment steps per seed |
| `--modes` | `dqn,noisynet,nrowan` | Comma list |
| `--seeds` | `0,1,2,3,4` | Comma list |
| `--eps-decay-steps` | `1000000` | DQN epsilon anneal horizon |

> The paper publishes a training budget only for CartPole (30K) and Pong (1M). For these two
> environments it is unstated, and at 30K **no method ever reaches the goal** - verified
> against a random policy, which solves 0 of 2000 MountainCar episodes. Acrobot reproduces
> at `--budget 200000`.

### 3. Continuous - MountainCarContinuous

```bash
# sparse regime (hard exploration) with a fair, non-decaying baseline
python main_mountaincar.py --modes nrowan_iso,vanilla --shaping off \
       --episodes 500 --noise-decay 1.0

# shaped regime
python main_mountaincar.py --shaping on

# with the optimizer fix
python main_mountaincar.py --modes nrowan_iso --shaping off --sigma-optimizer sgd
```

| Flag | Default | Notes |
|---|---|---|
| `--modes` | `nrowan,nrowan_iso,vanilla` | |
| `--shaping` | `on` | `off` = pure sparse reward. `on` adds potential-based shaping (`sin(3·pos) + 100·vel²`), provably policy-invariant and applied identically to every method; metrics always report the **true** env reward and success rate |
| `--episodes` | `150` | |
| `--seeds` | `0,1,2` | |
| `--noise-decay` | `0.99` | Per-episode decay of the **baseline's** Gaussian action noise. `1.0` = no decay. Ignored by the nrowan modes |
| `--sigma-optimizer`, `--sigma-floor` | see above | |

> **Set `--noise-decay 1.0` for any comparison you intend to report.** At the default of
> 0.99 the baseline's exploration noise reaches its 0.02 floor by roughly episode 230, so
> it explores with 2% of the action range while the noisy methods keep theirs at full
> strength - an unfair comparison a reviewer will notice.

### 4. Continuous - PointMaze

```bash
python main_pointmaze.py --noise-decay 1.0                       # all three variants
python main_pointmaze.py --modes nrowan_iso --sigma-optimizer sgd --noise-decay 1.0
python main_pointmaze.py --noise-decay 1.0 --resume              # continue after a drop
```

| Flag | Default |
|---|---|
| `--modes` | `nrowan,nrowan_iso,vanilla` |
| `--episodes` | `200` |
| `--seeds` | `0,1,2` |
| `--noise-decay` | `0.99` |
| `--sigma-optimizer`, `--sigma-floor` | see above |
| `--resume` | off - skip and reuse any `(mode, seed)` already checkpointed |

Environment: `PointMaze_Medium-v3`, an 8×8 maze of corridors and dead ends, 600-step cap,
fixed goal, sparse reward.

### 5. Continuous - CityLearn

```bash
python main_citylearn.py --smoke                     # 2-minute sanity check -- run this first
python main_citylearn.py --noise-decay 1.0           # full comparison
python main_citylearn.py --noise-decay 1.0 --resume  # continue after a drop
```

| Flag | Default | Notes |
|---|---|---|
| `--schema` | `citylearn_challenge_2022_phase_1` | 5 buildings, 5-D action, 44-D state |
| `--modes` | `nrowan,nrowan_iso,vanilla` | |
| `--episodes` | `60` | |
| `--episode-steps` | `1000` | `0` = the dataset's full length (8760 = one year) |
| `--seeds` | `0,1,2` | |
| `--noise-decay` | `0.99` | |
| `--sigma-optimizer`, `--sigma-floor` | see above | |
| `--resume`, `--smoke` | off | |

Two environment-side adaptations, applied identically to every method: **running
observation standardization** (raw features span four orders of magnitude - without this the
network does not converge at all) and **action rescaling** from the actor's `[-1, 1]` range
onto each dimension's own bounds.

> **Seasonal aliasing.** The dataset covers a year. With `--episode-steps 1000`, consecutive
> episodes cover different parts of the year and the reward oscillates with a period of about
> 8 episodes (autocorrelation 0.99). All methods are measured on the identical window so the
> comparison is unaffected, but prefer an evaluation window that is a multiple of 8 - or set
> `--episode-steps 0` to avoid the effect entirely.

### 6. Continuous - grid2op

```bash
python main.py                       # all three variants
python main.py --modes nrowan,vanilla
```

Only `--modes`. Configuration is fixed in the file: `rte_case14_realistic`, 150 episodes
per seed, 3 seeds, 2000-step cap, 5000 warm-up steps, batch 128. Very slow - budget
several hours per mode.

---

## Output

Every script writes to `results_<env>/`, and to Google Drive automatically when one is
mounted (`/content/drive/MyDrive/NROWAN_DDPG_Project/`), so a Colab disconnect does not
lose finished work.

- `run_<mode>[<tag>]_seed<N>.npz` - per-seed checkpoint, written **as soon as that seed
  finishes**. An interruption costs at most one seed. `--resume` reuses these.
- `<metric>_<mode>[<tag>].txt` - per-metric arrays, shape `[seeds, episodes]`
- `comparison_<env>.png` - learning curves with mean ± std bands

The continuous scripts also record the **output-layer sigma** each episode and print
`init / max / final` per mode. This is the key mechanism measurement - it is what shows
whether the exploration noise survived training.

Inspect partial results mid-run from a second notebook:

```python
from google.colab import drive; drive.mount('/content/drive')
import numpy as np, glob, os
d = '/content/drive/MyDrive/NROWAN_DDPG_Project/results_pointmaze'
for f in sorted(glob.glob(os.path.join(d, 'run_*.npz'))):
    with np.load(f) as z:
        r, s = z['rewards'], z['sigma']
    print(f"{os.path.basename(f)[4:-4]:28s} eps={len(r):3d}  "
          f"first10={r[:10].mean():8.1f}  last10={r[-10:].mean():8.1f}  "
          f"sigma {s[0]:.4f} -> {s[-1]:.4f}")
```

---

## Results summary

### Discrete reproduction - succeeded

CartPole, 5 instances × 64 evaluation rounds (the paper's Table 3 protocol):

| Method | Ours | Paper |
|---|---|---|
| DQN | 180.32 ± 20.74 | 170.49 ± 35.86 |
| NoisyNet-DQN | 174.46 ± 7.35 | 164.96 ± 31.56 |
| **NROWAN-DQN** | **191.72 ± 5.95** | **187.04 ± 13.99** |

Acrobot, same protocol, 200K-step budget:

| Method | Ours | Paper |
|---|---|---|
| DQN | -100.61 ± 32.48 | -87.24 ± 22.33 |
| NoisyNet-DQN | -88.14 ± 53.07 | -86.57 ± 29.32 |
| **NROWAN-DQN** | **-82.32 ± 19.69** | **-84.41 ± 15.58** |

The exact rank order reproduces in both environments, NROWAN has the lowest standard
deviation in both, and even the paper's own reported anomaly reproduces - in CartPole,
NoisyNet falls *below* plain DQN, because this environment punishes noisy actions.

### Continuous transfer

| Environment | Reward | Outcome |
|---|---|---|
| grid2op | dense | Neither method surpassed a do-nothing policy (1112 steps, 4.2 violations). Lever mismatch: blackouts are cascading line overloads, which need *discrete* topology control |
| CityLearn | dense | Statistical tie. vanilla -3118.2 ± 6.0, nrowan -3127.3 ± 11.8, nrowan_iso -3130.9 ± 14.1 - a 12.7 gap on a 687 improvement. The plain baseline also has the *lowest* variance |
| PointMaze | sparse | vanilla 14.4% ± 4.2, nrowan 14.4% ± 6.8, **nrowan_iso 20.0% ± 2.7** - nominally best, but within overlapping variance at 3 seeds |
| MountainCar | sparse | vanilla **0 of 3 seeds** even with undecayed exploration noise; nrowan_iso solved 1 of 3 (33.3% ± 47.1), converging to a 69-step route |

### The decisive experiment

PointMaze, one variable changed - does preserving the exploration noise help?

| Variant | Final sigma | Success rate |
|---|---|---|
| NROWAN isolated | 0.0000 (dead by episode 20) | 20.0% ± 2.7 |
| NROWAN isolated + `--sigma-optimizer sgd` | **0.0842 (preserved)** | **13.3% ± 2.7** |

Preserving the noise made performance **worse**. The exploration explanation for the
isolated variant's advantage is ruled out. In these continuous tasks the learned parameter
noise is a liability, and the paper's penalty - which destroys it quickly - happens to do
useful work, for the opposite reason to the one it was designed for.

---

## Reproduction notes

Four things the paper does not publish, which we reverse-engineered:

1. **CartPole is the 200-step-cap version.** Sec. 5.1 describes the episode ending at total
   reward "+200".
2. **Budgets are in environment steps, not episodes** - 30K for CartPole, 1M for Pong
   (Table 1). Budgets for MountainCar and Acrobot are never stated.
3. **Table 3 scores are post-training evaluation**, not training-curve averages: Sec. 5.3 -
   "we trained five instances for each algorithm, and each instance ran 64 rounds."
4. **The DQN epsilon schedule is never stated.** Mnih et al. (2015) taken *literally* is what
   matched: anneal 1.0 → 0.1 over 1M frames - so epsilon stays ≈1.0 for the entire 30K
   budget and DQN learns almost purely off-policy - with evaluation at epsilon 0.05. This
   single change moved our DQN from 199.9 to 180.3.

What we had to adapt when porting to continuous control:

| Component | Paper | Our port |
|---|---|---|
| Penalty `D` | Sum of `|sigma|`, output layer only | Unchanged |
| Noisy layers | Last two fully-connected layers | Unchanged |
| When `k` updates | Every step, from the running within-episode reward | Once per episode |
| How `k` is normalized | Against reward bounds known in advance (eq. 12) | Sparse tasks: bounds `[0,1]` on the success indicator. Grid and energy tasks: online min–max, because no reward bounds exist there |

---

## Caveats

- Every continuous result rests on **three seeds**. Where standard deviations overlap
  (PointMaze, CityLearn) we report a tie, not a winner.
- The **non-isolated** variant was not run on sparse MountainCar under identical settings,
  so the win there cannot be attributed to gradient isolation specifically.
- **grid2op agent scores predate a correctness fix** in the noise implementation and are not
  reported. The do-nothing reference, which never invokes the agent, is unaffected.
- The isolated variant's edge is plausibly reduced gradient variance, but with three seeds
  that cannot be separated from chance. We do not claim causality.
- `main_deceptive.py` (a 1-D deceptive corridor) was written but never run. It is kept only
  as a reserve testbed.
