import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        # Define the layers of the Actor network
        # layer 1: Input state -> Hidden layer of 256 neurons
        self.fc1 = nn.Linear(state_dim, 256)
        # layer 2: 256 neurons -> 128 neurons
        self.fc2 = nn.Linear(256, 128)
        # layer 3 (Output): 128 neurons -> action_dim (continuous action vector)
        self.fc3 = nn.Linear(128, action_dim)
        
        # Store the maximum possible action value (to scale the output)
        self.max_action = max_action

    def forward(self, state):
        # Pass state through the network with ReLU activation functions
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Use Tanh on the output layer to bound actions between -1 and +1
        x = torch.tanh(self.fc3(x))
        # Scale the action to the environment's acceptable range
        return self.max_action * x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # The Critic takes BOTH the state and the action as input
        # So the input size is state_dim + action_dim
        input_dim = state_dim + action_dim
        
        # Define the layers of the Critic network
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        # The output is a single number representing the Q-value (Quality of the action)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        # Concatenate the state and action vectors before feeding into the network
        x = torch.cat([state, action], dim=1)
        # Pass through the network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Return the estimated Q-value (no activation function on the final output)
        return self.fc3(x)