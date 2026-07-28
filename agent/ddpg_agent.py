import copy
import torch
import torch.nn.functional as F
import numpy as np
from agent.networks import Actor, Critic
from agent.noise import OnlineWeightAdjuster


class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001,
                 sigma_init=0.5, xi_max=1.0, mode="nrowan",
                 expl_noise=0.2, expl_noise_decay=0.99, expl_noise_min=0.02,
                 xi_inf_R=None, xi_sup_R=None,
                 sigma_optimizer="adam", sigma_floor=0.0):
        """
        mode = "nrowan"     -> our method: NoisyLinear actor + noise-reduction
                               loss D + online weight adjustment (exploration
                               inside the net). The policy loss is computed
                               through the NOISY forward pass, so sigma receives
                               gradients from BOTH the policy loss and D.
        mode = "nrowan_iso" -> gradient-isolated variant: identical to "nrowan"
                               EXCEPT the policy loss is computed through the
                               mean-only (noise-free) forward pass, so sigma is
                               invisible to the policy gradient and receives its
                               gradient from the D penalty ALONE. This removes
                               the mechanism by which Q-maximization suppresses
                               the exploration noise (sigma), making the learned
                               parameter noise as suppression-proof as vanilla's
                               external action noise.
        mode = "vanilla"    -> classic DDPG baseline: deterministic actor +
                               decaying Gaussian action-space noise, no D, no
                               online weight

        Two independent knobs on how sigma is allowed to decay (both default to
        the original behavior, so existing results stay reproducible):

        sigma_optimizer = "adam" -> original. Adam normalizes each gradient by
                               its own running magnitude (step ~ lr * g/|g|),
                               so the D penalty's tiny but perfectly consistent
                               downward gradient still moves sigma a full lr per
                               step. Measured: sigma dies within ~440 updates
                               (< 1 episode) regardless of how small xi is.
        sigma_optimizer = "sgd"  -> the fix: sigma parameters get a plain SGD
                               optimizer (step = lr * g, NOT normalized) while
                               mu and the rest keep Adam. Sigma's decay becomes
                               proportional to xi again, i.e. the gradual,
                               competence-gated annealing the paper intended.

        sigma_floor = 0.0    -> original: sigma may reach exactly 0, after which
                               it can never recover (the |sigma| gradient is 0
                               there and nothing pushes it back up).
        sigma_floor > 0      -> clamp every sigma to at least this value after
                               each update, guaranteeing a residual amount of
                               exploration noise for the whole run.
        """
        assert mode in ("nrowan", "nrowan_iso", "vanilla")
        assert sigma_optimizer in ("adam", "sgd")
        self.mode = mode
        self.sigma_floor = float(sigma_floor)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = discount
        self.tau = tau

        noisy = (mode != "vanilla")
        self.actor = Actor(state_dim, action_dim, max_action,
                           sigma_init=sigma_init, noisy=noisy).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        # Optionally split sigma off onto a NON-normalizing optimizer, so the D
        # penalty anneals it proportionally instead of at a fixed lr per step.
        self.sigma_params = [p for n, p in self.actor.named_parameters()
                             if 'sigma' in n] if noisy else []
        if noisy and sigma_optimizer == "sgd" and self.sigma_params:
            other = [p for n, p in self.actor.named_parameters() if 'sigma' not in n]
            self.actor_optimizer = torch.optim.Adam(other, lr=1e-4)
            self.sigma_optimizer = torch.optim.SGD(self.sigma_params, lr=1e-4)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
            self.sigma_optimizer = None

        # Critic (faster learning rate to guide the actor)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # NROWAN online weight adjustment for the noise-reduction loss D.
        # With xi_inf_R/xi_sup_R given, uses the paper's eq. 12 (known fixed
        # bounds); otherwise online min/max normalization.
        self.online_adjuster = OnlineWeightAdjuster(xi_max=xi_max,
                                                    inf_R=xi_inf_R, sup_R=xi_sup_R)
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

        if self.mode != "vanilla":
            # nrowan / nrowan_iso: exploration lives in the noisy weights. For
            # acting we use the BEHAVIORAL noise (frozen for the whole episode
            # -> coherent, directed exploration). eval() -> means only.
            if explore:
                self.actor.train()
                self.actor.set_behavioral(True)
            else:
                self.actor.eval()
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
        """NROWAN: sample one fresh BEHAVIORAL perturbation for the actor at the
        START of each episode (coherent, directed exploration). No-op for vanilla."""
        if self.mode != "vanilla":
            self.actor.train()
            self.actor.reset_behavioral_noise()

    def update_noise_weight(self, episode_reward):
        """Per-episode update. NROWAN modes: recompute xi from recent
        performance. Vanilla: decay the Gaussian action-noise std."""
        if self.mode != "vanilla":
            self.noise_weight = self.online_adjuster.update(episode_reward)
        else:
            self.expl_noise = max(self.expl_noise_min,
                                  self.expl_noise * self.expl_noise_decay)
        return self.noise_weight

    def _apply_sigma_floor(self):
        """Keep every sigma at or above sigma_floor. Also prevents sigma from
        crossing into negative values, where |sigma| would start pushing it
        further away from zero."""
        if self.sigma_floor <= 0.0 or not self.sigma_params:
            return
        with torch.no_grad():
            for p in self.sigma_params:
                p.clamp_(min=self.sigma_floor)

    def noise_magnitude(self):
        """Current output-layer noise level (mean |sigma|), for diagnostics.
        Should RISE while exploring, then anneal as the policy stabilizes."""
        return self.actor.output_sigma()

    def train(self, replay_buffer, batch_size=256):
        # Noisy layers must be in training mode so sigma gradients flow
        self.actor.train()
        self.critic.train()

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # ---------------------- CRITIC UPDATE ---------------------- #
        with torch.no_grad():
            # Fresh TRAINING noise on the target actor each step (NoisyNet).
            self.actor_target.set_behavioral(False)
            self.actor_target.reset_noise()
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)

            # Bellman Equation
            target_Q = reward + (1 - done) * self.discount * target_Q

        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # --- PROTECTION A: Gradient Clipping for Critic ---
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ---------------------- ACTOR UPDATE ---------------------- #
        if self.mode == "nrowan_iso":
            # GRADIENT ISOLATION: compute the policy loss through the MEAN-ONLY
            # forward pass (eval mode -> weight = mu, sigma absent from the
            # graph). mu still learns to maximize Q as usual, but sigma is
            # invisible to the policy gradient, so Q-maximization cannot
            # suppress the exploration noise. Sigma's ONLY gradient source is
            # the D penalty below, scheduled by the online weight xi.
            self.actor.eval()
            policy_loss = -self.critic(state, self.actor(state)).mean()
            self.actor.train()
        else:
            # Original NROWAN transfer: freshly resampled TRAINING noise (NOT
            # the frozen behavioral noise) so the policy loss is an expectation
            # over noise and sigma receives a policy gradient as well. The
            # per-episode behavioral noise used for acting is left untouched
            # -> coherent exploration AND learnable sigma coexist.
            self.actor.set_behavioral(False)
            self.actor.reset_noise()
            policy_loss = -self.critic(state, self.actor(state)).mean()

        # NROWAN noise-reduction loss D, weighted by the online xi
        noise_loss = self.actor.noise_loss()
        actor_loss = policy_loss + self.noise_weight * noise_loss

        self.actor_optimizer.zero_grad()
        if self.sigma_optimizer is not None:
            self.sigma_optimizer.zero_grad()
        actor_loss.backward()

        # --- PROTECTION A: Gradient Clipping for Actor ---
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        if self.sigma_optimizer is not None:
            self.sigma_optimizer.step()
        self._apply_sigma_floor()

        # ------------------- TARGET NETWORKS UPDATE ------------------- #
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()
