"""
Q-networks for NROWAN-DQN reproduction (discrete action spaces).

Reuses the SAME NoisyLinear block as the DDPG experiments, so the NROWAN
mechanism (learned per-weight sigma + noise-reduction loss D on the output
layer) is identical across the discrete and continuous studies.

Two heads:
  * MLPQNet  -> classic control (CartPole, Acrobot, MountainCar): 2 hidden
               layers of 128 units (paper Table 1, "Others").
  * CNNQNet  -> Atari (Pong): 3 conv layers (32/64/64, 8x8/4x4/3x3,
               stride 4/2/1) + 512-unit hidden (paper Table 1, "Pong").

In noisy modes the last two fully-connected layers are NoisyLinear; the
noise-reduction loss D is computed on the OUTPUT layer only (paper Sec. 4.1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.networks import NoisyLinear


class MLPQNet(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=128, sigma_init=0.4, noisy=True):
        super().__init__()
        self.noisy = noisy
        self.fc1 = nn.Linear(state_dim, hidden)
        if noisy:
            self.fc2 = NoisyLinear(hidden, hidden, sigma_init=sigma_init)
            self.fc3 = NoisyLinear(hidden, n_actions, sigma_init=sigma_init)
        else:
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)               # Q-values, one per action

    def reset_noise(self):
        if self.noisy:
            self.fc2.reset_noise()
            self.fc3.reset_noise()

    def noise_D(self):
        """NROWAN noise-reduction loss D on the OUTPUT layer only (paper eq. 8)."""
        if self.noisy:
            return self.fc3.noise_D()
        return torch.zeros((), device=self.fc1.weight.device)

    def output_sigma(self):
        """Mean |sigma| of the output layer (diagnostic)."""
        if self.noisy:
            return float(self.fc3.noise_magnitude().item())
        return 0.0


class CNNQNet(nn.Module):
    """Atari head. Input: (N, 1, 84, 84) grayscale (paper uses 1 frame, no stack)."""

    def __init__(self, n_actions, in_channels=1, sigma_init=0.4, noisy=True):
        super().__init__()
        self.noisy = noisy
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        conv_out = 64 * 7 * 7            # for 84x84 input
        if noisy:
            self.fc1 = NoisyLinear(conv_out, 512, sigma_init=sigma_init)
            self.fc2 = NoisyLinear(512, n_actions, sigma_init=sigma_init)
        else:
            self.fc1 = nn.Linear(conv_out, 512)
            self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def reset_noise(self):
        if self.noisy:
            self.fc1.reset_noise()
            self.fc2.reset_noise()

    def noise_D(self):
        if self.noisy:
            return self.fc2.noise_D()
        return torch.zeros((), device=self.conv1.weight.device)

    def output_sigma(self):
        if self.noisy:
            return float(self.fc2.noise_magnitude().item())
        return 0.0
