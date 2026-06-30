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

        # TRAINING noise buffers — resampled on every gradient step so that the
        # noise-reduction objective is a proper expectation over noise E[L] and
        # sigma actually receives a learning signal (standard NoisyNet behavior).
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        # BEHAVIORAL noise buffers — resampled ONCE per episode and held fixed
        # during action selection, giving coherent (directed) exploration across
        # the whole trajectory. Kept separate from the training noise so that
        # coherent behavior does NOT prevent sigma from being learned.
        self.register_buffer("weight_epsilon_b", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon_b", torch.empty(out_features))
        self.use_behavioral = False   # which noise the forward pass uses

        self.reset_parameters()
        self.reset_noise()
        self.reset_behavioral_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        # Initial sigma scaled by fan-in (in_features) for BOTH weight and bias,
        # as in Fortunato et al. (NoisyNet).
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Resample the TRAINING noise. Called on every gradient step so sigma
        gets a proper expectation-over-noise gradient."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def reset_behavioral_noise(self):
        """Resample the BEHAVIORAL noise (once per episode) for coherent
        exploration during action selection."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon_b.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon_b.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            # Noisy weights — exploration is ON and sigma gradients flow.
            # Behavioral (frozen, per-episode) noise during acting; training
            # (freshly resampled) noise during gradient updates.
            if self.use_behavioral:
                w_eps, b_eps = self.weight_epsilon_b, self.bias_epsilon_b
            else:
                w_eps, b_eps = self.weight_epsilon, self.bias_epsilon
            weight = self.weight_mu + self.weight_sigma * w_eps
            bias = self.bias_mu + self.bias_sigma * b_eps
        else:
            # Deterministic (evaluation): use the learned means only
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def noise_D(self):
        """Paper eq. (8): normalized total |sigma| of this layer,
        D = (sum|sigma_w| + sum|sigma_b|) / ((p* + 1) * Na)."""
        denom = (self.in_features + 1) * self.out_features
        return (self.weight_sigma.abs().sum() + self.bias_sigma.abs().sum()) / denom

    def noise_magnitude(self):
        """Mean absolute sigma of this layer (diagnostic for tracking sigma)."""
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
        """Resample TRAINING noise on the noisy layers (per gradient step)."""
        if self.noisy:
            self.fc2.reset_noise()
            self.fc3.reset_noise()

    def reset_behavioral_noise(self):
        """Resample BEHAVIORAL noise on the noisy layers (once per episode)."""
        if self.noisy:
            self.fc2.reset_behavioral_noise()
            self.fc3.reset_behavioral_noise()

    def set_behavioral(self, flag):
        """Select which noise the forward pass uses: behavioral (acting) or
        training (gradient updates)."""
        if self.noisy:
            self.fc2.use_behavioral = flag
            self.fc3.use_behavioral = flag

    def noise_loss(self):
        """NROWAN noise-reduction loss D. Per the paper (eq. 8 + Sec. 4.1), the
        penalty is applied to the OUTPUT layer ONLY -- the hidden layer's sigma
        is deliberately left free to stay large. Returns 0 for the vanilla
        (non-noisy) baseline."""
        if self.noisy:
            return self.fc3.noise_D()
        return torch.zeros((), device=self.fc1.weight.device)

    def output_sigma(self):
        """Mean |sigma| of the OUTPUT layer (diagnostic): should rise while the
        agent explores, then anneal as the policy stabilizes."""
        if self.noisy:
            return float(self.fc3.noise_magnitude().item())
        return 0.0


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
