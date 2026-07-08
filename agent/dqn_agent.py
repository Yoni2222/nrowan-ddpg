"""
NROWAN-DQN agent for the faithful paper reproduction (discrete actions).

Supports the paper's three algorithms via `mode`:
  * "dqn"      -> plain DQN, epsilon-greedy exploration, no noise.
  * "noisynet" -> NoisyNet-DQN: noisy layers for exploration, NO noise
                  reduction (k = 0).
  * "nrowan"   -> NoisyNet + noise-reduction loss D + online weight k.

Faithful to the paper (Sec. 4, Algorithm 1, Tables 1-2):
  * Loss = TD-error + k * D, with D the output-layer noise (eq. 8).
  * k computed online from the WITHIN-EPISODE cumulative reward (eq. 12):
        k = k_final * (cum_r - inf_R) / (sup_R - inf_R),  reset each episode.
  * Hard target update every `target_update` steps.
  * NoisyNet noise resampled every forward pass (acting AND each update).
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from agent.q_networks import MLPQNet, CNNQNet


class DQNAgent:
    def __init__(self, state_dim, n_actions, mode="nrowan", arch="mlp",
                 lr=1e-3, gamma=0.99, target_update=1000,
                 sigma_init=0.4, k_final=4.0, inf_R=0.0, sup_R=1.0,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=10000,
                 in_channels=1):
        assert mode in ("dqn", "noisynet", "nrowan")
        assert arch in ("mlp", "cnn")
        self.mode = mode
        self.gamma = gamma
        self.target_update = target_update
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        noisy = mode in ("noisynet", "nrowan")
        if arch == "mlp":
            self.q = MLPQNet(state_dim, n_actions, sigma_init=sigma_init, noisy=noisy)
        else:
            self.q = CNNQNet(n_actions, in_channels=in_channels,
                             sigma_init=sigma_init, noisy=noisy)
        self.q = self.q.to(self.device)
        self.q_target = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

        # Online weight (eq. 12) — only active in nrowan mode
        self.k_final = k_final
        self.inf_R = inf_R
        self.sup_R = sup_R
        self.k = 0.0
        self.cum_r = 0.0

        # epsilon-greedy schedule — only used in dqn mode
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        self.total_steps = 0     # env steps taken (drives epsilon)
        self.train_steps = 0     # gradient updates (drives target sync)

    # ----------------------------- acting ----------------------------- #
    def epsilon(self):
        frac = max(0.0, 1.0 - self.total_steps / self.eps_decay_steps)
        return self.eps_end + (self.eps_start - self.eps_end) * frac

    def select_action(self, state, explore=True):
        self.total_steps += 1
        if self.mode == "dqn":
            if explore and np.random.rand() < self.epsilon():
                return np.random.randint(self.n_actions)
            self.q.eval()
        else:
            # noisynet / nrowan: exploration comes from the weight noise itself
            if explore:
                self.q.train()
                self.q.reset_noise()
            else:
                self.q.eval()

        with torch.no_grad():
            s = torch.FloatTensor(np.asarray(state)).unsqueeze(0).to(self.device)
            return int(self.q(s).argmax(dim=1).item())

    # ------------------------- online weight k ------------------------ #
    def update_k_step(self, reward):
        """Per-step update of the online weight from within-episode cumulative
        reward (paper eq. 12). No-op except in nrowan mode."""
        self.cum_r += reward
        if self.mode == "nrowan":
            spread = self.sup_R - self.inf_R
            if spread > 1e-8:
                prog = (self.cum_r - self.inf_R) / spread
                self.k = self.k_final * float(np.clip(prog, 0.0, 1.0))
            else:
                self.k = 0.0

    def end_episode(self):
        self.cum_r = 0.0     # r+_t reset at terminal (Algorithm 1, line 12)

    # ------------------------------ learn ----------------------------- #
    def train(self, buffer, batch_size=32):
        self.q.train()
        state, action, reward, next_state, done = buffer.sample(batch_size)
        action = action.long()                       # discrete indices [B,1]

        self.q.reset_noise()                         # fresh noise for this update
        q = self.q(state).gather(1, action)

        with torch.no_grad():
            if self.mode != "dqn":
                self.q_target.train()
                self.q_target.reset_noise()
            else:
                self.q_target.eval()
            max_next = self.q_target(next_state).max(dim=1, keepdim=True)[0]
            target = reward + (1.0 - done) * self.gamma * max_next

        td_loss = F.mse_loss(q, target)
        loss = td_loss + self.k * self.q.noise_D()   # k=0 for dqn/noisynet

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return td_loss.item()

    def noise_magnitude(self):
        return self.q.output_sigma()
