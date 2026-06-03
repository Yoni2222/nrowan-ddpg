import copy
import torch
import torch.nn.functional as F
import numpy as np
from agent.networks import Actor, Critic
from agent.noise import NROWANParameterNoise

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = discount
        self.tau = tau

        # Initialize Actor
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        # Initialize Critic
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # Initialize NROWAN Parameter Noise Mechanism
        # The initial_std dictates how aggressive the exploration is at the start
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
                # 1. Corrupt the Actor's weights with noise
                self.noise_model.apply_noise(self.actor)
                
            # 2. Forward pass through the network (either noisy or clean)
            action = self.actor(state).cpu().data.numpy().flatten()
            
            if explore:
                # 3. Immediately restore the clean weights so training isn't ruined
                self.noise_model.revert_noise(self.actor)

            # Clip the action to valid bounds
            return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=256):
        """
        Sample a batch from memory and update the Actor and Critic weights.
        """
        # 1. Sample a random mini-batch of experiences from the replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # ---------------------- CRITIC UPDATE ---------------------- #
        # Compute the target Q value (what the Critic SHOULD predict)
        with torch.no_grad():
            # Get next action from the Target Actor
            next_action = self.actor_target(next_state)
            # Get target Q-value from the Target Critic
            target_Q = self.critic_target(next_state, next_action)
            # Bellman equation: Target = Reward + Gamma * Target_Q * (1 - done)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Get current Q estimate from the main Critic
        current_Q = self.critic(state, action)

        # Compute Critic loss (Mean Squared Error between current Q and target Q)
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the Critic (Backpropagation)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------- ACTOR UPDATE ---------------------- #
        # Compute Actor loss 
        # We want to MAXIMIZE the Critic's evaluation, so we MINIMIZE the negative Critic evaluation
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the Actor (Backpropagation)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------- TARGET NETWORKS UPDATE ------------------- #
        # Soft update of the frozen target models (Polyak averaging)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)