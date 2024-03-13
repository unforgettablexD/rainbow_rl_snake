import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingNetwork(nn.Module):
    def __init__(self, input_dims, num_actions, atom_size=51):
        super(DuelingNetwork, self).__init__()

        self.input_dims = input_dims
        self.num_actions = num_actions
        self.atom_size = atom_size

        # Common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(*self.input_dims, 128),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.atom_size)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions * self.atom_size)
        )

        # Support for the distributional DQN
        self.register_buffer("support", torch.linspace(-10, 10, self.atom_size))

    def forward(self, state):
        features = self.feature_layer(state)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        advantage = advantage.view(-1, self.num_actions, self.atom_size)
        value = value.view(-1, 1, self.atom_size)

        # Combine streams
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Apply softmax to get probability distribution
        dist = F.softmax(q_values, dim=-1)

        q_values = torch.sum(dist * self.support, dim=2)

        return q_values, dist

    def calculate_q_values(self, distributions):
        """ Calculate the Q values from the distributions for decision making """
        q_values = torch.sum(distributions * self.support, dim=2)
        return q_values


