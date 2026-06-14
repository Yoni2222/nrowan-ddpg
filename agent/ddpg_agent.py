import copy
import torch
import torch.nn.functional as F
import numpy as np
from agent.networks import Actor, Critic
from agent.noise import OnlineWeightAdjuster


class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001,
                 sigma_init=0.5, xi_max=1.0, mode="nrowan",
                 expl_noise=0.2, expl_noise_decay=0.99, expl_noise_min=0.02):
        """
        mode = "nrowan"  -> our method: NoisyLinear actor + noise-reduction loss D
                            + online weight adjustment (exploration inside the net)
        mode = "vanilla" -> classic DDPG baseline: deterministic actor + decaying
                            Gaussian action-space noise, no D, no online weight
        """
        assert mode in ("nrowan", "vanilla")
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = discount
        self.tau = tau

        noisy = (mode == "nrowan")
        self.actor = Actor(state_dim, action_dim, max_action,
                           sigma_init=sigma_init, noisy=noisy).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        # Critic (faster learning rate to guide the actor)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # NROWAN online weight adjustment for the noise-reduction loss D
        self.online_adjuster = OnlineWeightAdjuster(xi_max=xi_max)
        self.noise_weight = 0.0   # xi, updated once per episode (nrowan only)

        # Vanilla-baseline Gaussian action-noise schedule (ignored in nrowan mode)
        self.expl_noise = expl_noise            # std as a fraction of max_action
        self.expl_noise_decay = expl_noise_decay
        self.expl_noise_min = expl_noise_min

        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state, explore=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        if self.mode == "nrowan":
            # Exploration lives in the noisy weights. train() -> noise active;
            # the noise is NOT resampled here (reset once per episode, see
            # reset_exploration_noise) so exploration is COHERENT across the
            # episode. eval() -> deterministic policy (learned means only).
            self.actor.train() if explore else self.actor.eval()
            with torch.no_grad():
                action = self.actor(state).cpu().data.numpy().flatten()
        else:
            # Vanilla DDPG: deterministic actor + external Gaussian action noise
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(state).cpu().data.numpy().flatten()
            if explore:
                action = action + np.random.normal(
                    0.0, self.expl_noise * self.max_action, size=action.shape)

        return np.clip(action, -self.max_action, self.max_action)

    def reset_exploration_noise(self):
        """NROWAN: sample one fresh exploration perturbation for the actor at the
        START of each episode (coherent, directed exploration). No-op for vanilla."""
        if self.mode == "nrowan":
            self.actor.train()
            self.actor.reset_noise()

    def update_noise_weight(self, episode_reward):
        """Per-episode update. NROWAN: recompute xi from recent performance.
        Vanilla: decay the Gaussian action-noise std."""
        if self.mode == "nrowan":
            self.noise_weight = self.online_adjuster.update(episode_reward)
        else:
            self.expl_noise = max(self.expl_noise_min,
                                  self.expl_noise * self.expl_noise_decay)
        return self.noise_weight

    def noise_magnitude(self):
        """Current NROWAN noise level D (sum of learned sigmas). Should trend
        DOWN over training as the policy becomes more deterministic."""
        with torch.no_grad():
            return self.actor.noise_loss().item()

    def train(self, replay_buffer, batch_size=256):
        # Noisy layers must be in training mode so sigma gradients flow
        self.actor.train()
        self.critic.train()

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # ---------------------- CRITIC UPDATE ---------------------- #
        with torch.no_grad():
            self.actor_target.reset_noise()
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)

            # Bellman Equation
            target_Q = reward + (1 - done) * self.discount * target_Q

            # --- PROTECTION C: Target Q-Value Clipping ---
            target_Q = torch.clamp(target_Q, min=-150.0, max=150.0)

        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # --- PROTECTION A: Gradient Clipping for Critic ---
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ---------------------- ACTOR UPDATE ---------------------- #
        # NOTE: we deliberately do NOT resample the actor's noise here. The
        # exploration epsilon is fixed per episode (reset_exploration_noise), so
        # training must not disturb it. sigma still gets gradients through the
        # fixed epsilon and through the noise-reduction loss D below.
        policy_loss = -self.critic(state, self.actor(state)).mean()

        # NROWAN noise-reduction loss D, weighted by the online xi
        noise_loss = self.actor.noise_loss()
        actor_loss = policy_loss + self.noise_weight * noise_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # --- PROTECTION A: Gradient Clipping for Actor ---
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # ------------------- TARGET NETWORKS UPDATE ------------------- #
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()
