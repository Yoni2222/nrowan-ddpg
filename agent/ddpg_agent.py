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

        # Initialize Actor (Polished learning rate: 1e-4 to avoid destabilizing the policy)
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        # Initialize Critic (Polished learning rate: 1e-3 so it learns faster than the actor)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Initialize NROWAN Parameter Noise Mechanism
        self.noise_model = NROWANParameterNoise(initial_std=0.1, decay_rate=0.998)

        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state, explore=True):
        """
        Selects an action based on the current state.
        Applies NROWAN Parameter Noise if explore is True.
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        if explore:
            self.noise_model.apply_noise(self.actor)
            
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if explore:
            self.noise_model.revert_noise(self.actor)

        return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=256):
        """
        Sample a batch from memory, update networks, and return loss statistics.
        """
        # 1. Sample from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # ---------------------- CRITIC UPDATE ---------------------- #
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.discount * target_Q

        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------- ACTOR UPDATE ---------------------- #
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------- TARGET NETWORKS UPDATE ------------------- #
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Return raw scalars back for tracking and diagnostic plotting
        return critic_loss.item(), actor_loss.item()