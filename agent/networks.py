import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    Factorized Gaussian NoisyLinear layer (Fortunato et al., 2018) — the core
    building block of NROWAN. The weights are parameterized as:

        w = mu + sigma * epsilon

    where `mu` and `sigma` are LEARNED parameters and `epsilon` is resampled
    noise. The agent therefore learns *how much* exploration noise to use per
    weight, and the NROWAN noise-reduction loss (sum of |sigma|) lets us push
    that noise down online as the policy stabilizes.
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable mean and standard-deviation parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not learned, resampled every reset_noise())
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        # Initial sigma scaled by fan-in, as in the NoisyNet paper
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Resample the factorized Gaussian noise. Call before each forward pass
        where fresh exploration / target noise is desired."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            # Noisy weights — exploration is ON and sigma gradients flow
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Deterministic (evaluation): use the learned means only
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def noise_magnitude(self):
        """Mean absolute sigma of this layer — the per-layer noise level used by
        the NROWAN noise-reduction loss D."""
        return self.weight_sigma.abs().mean() + self.bias_sigma.abs().mean()


class Actor(nn.Module):
    """
    DDPG Actor with NROWAN noisy layers on the head. The input layer stays a
    plain Linear (keeps the state encoding stable); the last two layers are
    NoisyLinear so exploration is driven by learned parameter-space noise
    instead of an external noise process.
    """

    def __init__(self, state_dim, action_dim, max_action, sigma_init=0.5, noisy=True):
        super(Actor, self).__init__()

        self.noisy = noisy
        self.fc1 = nn.Linear(state_dim, 256)
        if noisy:
            # NROWAN: learned-noise head
            self.fc2 = NoisyLinear(256, 128, sigma_init=sigma_init)
            self.fc3 = NoisyLinear(128, action_dim, sigma_init=sigma_init)
        else:
            # Vanilla DDPG baseline: plain deterministic head
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Tanh bounds the action to [-1, 1]; scaled by max_action
        x = torch.tanh(self.fc3(x))
        return self.max_action * x

    def reset_noise(self):
        if self.noisy:
            self.fc2.reset_noise()
            self.fc3.reset_noise()

    def noise_loss(self):
        """NROWAN noise-reduction loss D: total noise magnitude across the
        noisy layers. Minimizing this drives the policy toward determinism.
        Returns 0 for the vanilla (non-noisy) baseline."""
        if self.noisy:
            return self.fc2.noise_magnitude() + self.fc3.noise_magnitude()
        return torch.zeros((), device=self.fc1.weight.device)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # The Critic takes BOTH the state and the action as input
        input_dim = state_dim + action_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        # Single Q-value output
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
