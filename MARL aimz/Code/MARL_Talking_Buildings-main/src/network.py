import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self,  input_side_len: int, output_size: int):
        super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_side_len, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(params=self.parameters(), lr=0.0001)

    def forward(self, input: torch.Tensor):
        output = self.relu(self.fc1(input))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)

        return output


class ActorCriticDiscreteNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCriticDiscreteNetwork, self).__init__()

        # Common network layers (shared between actor and critic)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        # Actor head (outputs action probabilities)
        self.actor = nn.Linear(128, action_dim)  # action_dim should be 10 (for 10 discrete actions)

        # Critic head (outputs the state value)
        self.critic = nn.Linear(128, 1)  # Outputs a single value for state value estimation

    def forward(self, state):
        # Forward pass through the common network layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Actor head (softmax to get action probabilities)
        action_probs = F.softmax(self.actor(x), dim=-1)

        # Critic head (state value estimation)
        state_value = self.critic(x)

        return action_probs, state_value
