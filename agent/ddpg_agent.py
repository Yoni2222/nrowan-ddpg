import copy
import torch
import torch.nn.functional as F
import numpy as np
from agent.networks import Actor, Critic
from agent.noise import NROWANParameterNoise

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = discount
        self.tau = tau

        # Initialize Actor (Stabilized learning rate: 1e-4)
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        # Initialize Critic (Faster learning rate: 1e-3 to let the teacher guide the student)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Initialize NROWAN Parameter Noise Mechanism
        self.noise_model = NROWANParameterNoise(initial_std=0.1, decay_rate=0.998)

        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state, explore=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        if explore:
            self.noise_model.apply_noise(self.actor)
        action = self.actor(state).cpu().data.numpy().flatten()
        if explore:
            self.noise_model.revert_noise(self.actor)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # ---------------------- CRITIC UPDATE ---------------------- #
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            
            # Bellman Equation
            target_Q = reward + (1 - done) * self.discount * target_Q
            
            # --- PROTECTION C: Target Q-Value Clipping ---
            # Prevents the values from spiraling into negative infinity due to continuous penalties
            target_Q = torch.clamp(target_Q, min=-150.0, max=150.0)

        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # --- PROTECTION A: Gradient Clipping for Critic ---
        # Prevents exploding gradients from corrupting network weights
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        
        self.critic_optimizer.step()

        # ---------------------- ACTOR UPDATE ---------------------- #
        actor_loss = -self.critic(state, self.actor(state)).mean()

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