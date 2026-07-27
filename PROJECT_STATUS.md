# NROWAN Project — Status & Handoff

**Last updated:** 2026-07-25
**Repo:** `C:\nrowan-ddpg\` (git repo, venv at `venv/Scripts/python.exe`)

---

## 1. Project goal

Master's RL course project requiring a novelty contribution, based on the paper:

> **NROWAN-DQN: A Stable Noisy Network with Noise Reduction and Online Weight
> Adjustment for Exploration** — Han, Zhou, Liu, Lü ([arXiv:2006.10980](https://arxiv.org/abs/2006.10980))

**The paper's mechanism (discrete, DQN):**
- `NoisyLinear` layers — each weight is `mu + sigma * eps`, with `mu`/`sigma` learned
- **Noise-reduction loss D** (eq. 8) — penalizes `|sigma|`, applied to the **output layer only**
- **Online weight k / xi** (eq. 12) — the weight of D, raised as the agent's
  performance improves, so exploration noise anneals only once competent

**Our novelty:** transfer this mechanism to a **continuous** action space (DDPG).

---

## 2. Part A — Reproducing the paper (discrete). STATUS: SUCCESS

### CartPole — reproduced, matches Table 3

5 seeds × 64 post-training evaluation rounds (the paper's protocol):

| method | ours | paper |
|---|---|---|
| DQN | 180.3 ± 20.7 | 170.5 ± 35.9 |
| NoisyNet-DQN | 176.5 ± 7.4 | 165.0 ± 31.6 |
| **NROWAN-DQN** | **191.7 ± 6.0** | **187.0 ± 14.0** |

Same ordering as the paper, NROWAN most stable — the paper's central claim reproduced.

### Acrobot — reproduced (seed 0 only, 200K-step budget)

| method | ours | paper |
|---|---|---|
| DQN | -87.08 ± 19.61 | -87.24 ± 22.33 |
| NoisyNet-DQN | -81.66 ± 76.05 | -86.57 ± 29.32 |
| **NROWAN-DQN** | **-73.36 ± 12.04** | **-84.41 ± 15.58** |

DQN baseline matches the paper to within 0.16 points. **Still needs the full 5 seeds.**

### Four things the paper does NOT state, which we reverse-engineered

1. **CartPole is the 200-step-cap version (v0), not v1's 500.** Paper Sec. 5.1 says
   the game ends at total reward "+200". Implemented via
   `gym.make("CartPole-v1", max_episode_steps=200)`.
2. **The budget is in environment steps, not episodes** — 30K for CartPole, 1M for
   Pong (Table 1). Budgets for MountainCar/Acrobot are **never published**.
3. **Table 3 scores are post-training evaluation**, not training-curve averages:
   Sec. 5.3 — "we trained five instances for each algorithm, and each instance ran
   64 rounds."
4. **The DQN epsilon schedule is never published.** Mnih et al. (2015) taken
   *literally* is what matched: anneal 1.0 → 0.1 over **1M frames** (so epsilon stays
   ≈1.0 for the entire 30K budget — DQN learns almost purely off-policy), eval
   epsilon 0.05. This single change moved our DQN from 199.9 → 180.3.

### Not reproduced

- **MountainCar-v0 (discrete):** all three algorithms score -200 at the 30K budget —
  nobody ever reaches the goal. Verified a random policy solves **0/2000** episodes.
  The paper's budget for this env is unpublished. Unresolved.
- **Pong:** `main_pong.py` is written and matches Table 1 exactly, but has **never
  been run**. Needs a GPU and ~4-8h per run × 15 runs (3 algorithms × 5 seeds).

---

## 3. Part B — Continuous extension (DDPG). STATUS: negative → diagnosed → first positive result

### The critical bug fix (2026-06-29)

Four bugs were found and fixed in the core files. **Any result produced before
2026-06-29 is suspect:**
- Noise was frozen during training → **sigma never learned at all** (the severe one)
- D was applied to all layers instead of the output layer only
- `bias_sigma` initialized with the wrong fan
- An ad-hoc ±150 clamp on target-Q

### The mechanism we discovered

In continuous control, `policy_loss = -Q(s, actor(s))`. Because the actor's noisy
forward pass makes the action a differentiable function of `sigma`, and noise lowers
`E[Q]` (Jensen, Q is locally concave), **the policy gradient actively suppresses
sigma** — Q-maximization finds that the cheapest way to raise Q is to turn the
exploration noise off.

Vanilla DDPG is immune: its Gaussian noise is added **outside** the computation
graph (plain NumPy in `select_action`), so no gradient can reach it.

**Measured evidence:**
- Original `nrowan`, shaped MountainCar: sigma **0.2652 → 0.1001**
- `nrowan_iso` (isolated), seeds that never succeeded: sigma **0.2652 → 0.2652 exactly**

### The fix: `nrowan_iso` mode (gradient isolation)

Compute `policy_loss` through the **mean-only** forward pass (`actor.eval()`, so
`weight = mu` and sigma is absent from the graph). `mu` still learns to maximize Q;
sigma's **only** gradient source becomes the D penalty, scheduled by the online weight.

Verified directly: policy-gradient norm on output-layer sigma is **0.234 under
`nrowan`** vs **exactly 0.000 under `nrowan_iso`**.

### Second fix: the online weight adjuster

`OnlineWeightAdjuster` normalized against a **running min/max**, so the *first
success ever* made the current average equal the running max → xi jumped straight to
`xi_max` → D crushed sigma to zero within episodes, killing exploration exactly when
the agent first found the goal. (Observed: seed 0's lone success at episode 91
collapsed sigma to ~0.)

Fixed by supporting the paper's original eq. 12 — normalization against **known fixed
bounds**. Success-gated scripts pass `[0, 1]`, so xi is proportional to the recent
success rate: one success in a 20-episode window → 5% of `xi_max`. grid2op keeps the
legacy online normalization (no known bounds there).

### Continuous results

| experiment | code | result |
|---|---|---|
| **grid2op** | OLD (buggy) | do-nothing 1112, vanilla 879, nrowan 465 — both agents worse than doing nothing |
| **MountainCar shaped**, 150 eps | FIXED | vanilla 91.2±1.2 / **97.8%** / 195 steps · nrowan -0.0 / **0.0%** / 999 |
| **MountainCar sparse**, 150 eps | FIXED, pre-xi-fix | nrowan_iso -7.0±2.8 / 0% · vanilla -0.3 / 0% |
| **MountainCar sparse, 500 eps** | FIXED + xi fix | **nrowan_iso 27.2±46.9 / 33.3%±47.1 / 689±438** · **vanilla -0.0±0.0 / 0.0% / 999** |
| **PointMaze Medium** | **UNCERTAIN** | nrowan 11.1±4.2% · vanilla 14.4±3.1% |

**The headline result (last row of MountainCar):** decoding the per-seed breakdown
from the mean/std arithmetic (the length std matches exactly: 438.4 vs 438 reported):
- **1 seed solves completely** — 100% success, reaching the goal in ~69 steps, reward ~+93
- 2 seeds never find the goal, and keep sigma at exactly 0.2652 (exploration preserved)
- Vanilla is **paralyzed in all 3 seeds** (reward -0.0 = the do-nothing local optimum
  produced by the -0.1·a² control cost)

This is the first time anything beat vanilla in a continuous environment.

### grid2op diagnosis (why it failed, and why the fix won't help there)

The failure is a **lever mismatch**, not an exploration problem: rte_case14 blackouts
are cascading line overloads, which need **discrete topology control** (bus splitting,
line switching), not continuous redispatch. Both agents were worse than do-nothing
because *any* action does net harm. Since our fix only *preserves noise for longer*,
it is predicted to make things worse there, not better. **Recommendation: report
grid2op as domain analysis in the writeup, not as a numeric comparison.** A full
rerun costs 10-12h for a predicted-negative result.

---

## 4. Honest caveats — do NOT overclaim these in the report

1. **1 seed out of 3 is suggestive, not conclusive.** Needs 8-10 seeds.
2. **The critical control has not been run.** The winning run changed *three* things
   at once (isolation + xi fix + 150→500 episodes). Without running the original
   `nrowan` at 500 episodes, we cannot attribute the success to gradient isolation.
3. **Baseline fairness problem.** Vanilla's exploration noise decays ×0.99/episode to
   a floor of 0.02, reaching that floor by ~episode 230 — so for the last 270 episodes
   it explores with 2% of the action range while our method keeps sigma at 0.2652.
   A reviewer will ask about this. Needs a no-decay vanilla control.
4. **The causal link between sigma decline and the 0% in shaped MountainCar is
   inference, not proof.** Sigma fell to 0.100, which is lower but not zero. Other
   explanations (e.g. `SIGMA_INIT=1.5` being too large and disrupting the policy)
   have not been excluded.
5. **PointMaze provenance is unknown.** File edit dates show when code was edited, not
   when Colab runs happened. It may have used the buggy pre-June-29 code. Needs rerun.

---

## 5. Code layout

### Shared agent code
| file | contents |
|---|---|
| `agent/networks.py` | `NoisyLinear` (dual noise buffers: behavioral=frozen per episode for acting, training=resampled per gradient step), `Actor`, `Critic` |
| `agent/q_networks.py` | `MLPQNet` (128×128, classic control), `CNNQNet` (Atari) |
| `agent/dqn_agent.py` | `DQNAgent` — modes `dqn` / `noisynet` / `nrowan` |
| `agent/ddpg_agent.py` | `DDPGAgent` — modes `nrowan` / `nrowan_iso` / `vanilla` |
| `agent/noise.py` | `OnlineWeightAdjuster` — fixed-bounds (eq. 12) or legacy online normalization |
| `agent/memory.py` | `ReplayBuffer`, `ImageReplayBuffer` (uint8 frames for Atari) |

### Run scripts
| file | env | flags |
|---|---|---|
| `main_cartpole.py` | CartPole (200-step cap) | — |
| `main_classic_dqn.py` | MountainCar-v0, Acrobot-v1 | `--env --budget --modes --seeds --eps-decay-steps` |
| `main_pong.py` | Pong (Atari) | `--modes --seeds --quick --summary` |
| `main.py` | grid2op | `--modes` |
| `main_mountaincar.py` | MountainCarContinuous-v0 | `--modes --shaping --episodes` |
| `main_pointmaze.py` | PointMaze_Medium-v3 | `--modes` |

### Flags discussed but NOT yet implemented
- `--seeds` for the continuous scripts (currently hardcoded `[0, 1, 2]`)
- `--noise-decay` to disable vanilla's exploration decay (fair-baseline control)

---

## 6. Pending work, in priority order

1. **The critical control** — `python main_mountaincar.py --modes nrowan --shaping off --episodes 500`
   Determines whether gradient isolation, or merely the xi fix + more episodes, caused the win.
2. **Fair baseline** — vanilla with no noise decay, same settings.
3. **More seeds** — 8-10 per method, to turn "1 of 3" into a real number.
4. **PointMaze** — rerun with all three modes on the current code.
5. **Acrobot** — full 5 seeds (currently only seed 0).
6. *(optional, expensive)* Pong; MountainCar-v0 discrete with a larger budget.

### Ideas considered and rejected
- **TD3** — same deterministic policy gradient, so the identical suppression occurs;
  would not fix anything. *Worth doing only as a generality check* ("the finding is
  not DDPG-specific"), ~50 lines on top of existing code.
- **SAC** — its entropy bonus explicitly rewards randomness, so it sidesteps the
  failure mode — but that also makes our mechanism partly redundant. Best used as a
  *conceptual contrast in the writeup*, not implemented.
- **PowerGym** — verified via its paper: the action space is **discrete** (capacitors
  binary, regulator taps integer, battery discharge integer), so it cannot test the
  continuous contribution.
- **grid2op with topology actions** — the right lever, but discrete, which defeats the
  purpose of testing a continuous extension.

---

## 7. Working constraints (important)

- **The user performs ALL git operations himself.** Never run any git command (not
  even `git status`). Provide commit message text only.
- **All heavy training runs on Google Colab**, not locally. Do not start long local runs.
- **The user writes in Hebrew and wants replies in Hebrew, without Latin letters in
  prose** (mixed direction scrambles the text). Code identifiers are unavoidable.
- The `Edit` tool works fine in `C:\nrowan-ddpg\` (it is not under OneDrive).
