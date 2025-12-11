from torch import optim

from building import Building
from network import ActorCriticDiscreteNetwork
import torch
import torch.nn as nn
import numpy as np
from agent import Agent
from environment import Env


class PPOAgent(Agent):
    def __init__(self, name, building: Building, includePriceInReward = True):
        super(PPOAgent, self).__init__(name, building)
        self.network = ActorCriticDiscreteNetwork(self.input_len, self.action_dim).to(self.device)  # Discrete network
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.0001)
        self.includePriceInReward = includePriceInReward

    def get_reward(self, env: Env):
        if self.building.office_hours[env.get_time()]:
            temp_penalty = -((self.building.wish_temp - self.building.temp) ** 2)
        else:
            temp_penalty = 0

        current_grid_price = env.grid.current_price
        current_consumption = self.building.consumption
        total_price = current_grid_price * current_consumption  #between 0 and 5600

        price_penalty = max(-(0.5*total_price), -50)

        if self.includePriceInReward:
            return temp_penalty + price_penalty
        else:
            return temp_penalty

    def take_action(self, observation: np.ndarray):
        observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        action_probs, value = self.network(observation)

        # Sample action from Categorical distribution
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)

        return action.item(), action.item() * self.kWh_step, log_prob.item(), value.item()

    def compute_advantages(self, next_value):
        advantages = []
        advantage = 0
        for t in reversed(range(self.memory_counter)):
            delta = self.reward_buffer[t] + self.gamma * next_value - self.value_buffer[t]
            advantage = delta + self.gamma * self.lamda * advantage
            advantages.insert(0, advantage)
            next_value = self.value_buffer[t]
        self.advantage_buffer[:self.memory_counter] = advantages

    def learn(self):
        if self.memory_counter < self.batch_size:
            return

        rewards = []
        discounted_reward = 0

        for reward in reversed(self.reward_buffer):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

            # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards_discounted = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        for _ in range(4):  # PPO typically updates for several epochs
            max_mem = min(self.memory_counter, self.memory_size)
            indices = np.random.choice(max_mem, self.batch_size, replace=False)

            states = torch.tensor(self.state_buffer[indices], dtype=torch.float32).to(self.device)
            actions = torch.tensor(self.action_buffer[indices], dtype=torch.int64).to(self.device)
            log_probs = torch.tensor(self.log_prob_buffer[indices], dtype=torch.float32).to(self.device)
            old_values = torch.tensor(self.value_buffer[indices], dtype=torch.float32).to(self.device)
            action_probs, values = self.network(states)

            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)

            # Compute ratios for the PPO loss
            ratios = torch.exp(new_log_probs - log_probs)

            advantage = rewards_discounted[indices].detach() - values.detach()  # advantage! can be changed to gae (#todo test)
            advantages = (advantage - advantage.mean()) / (advantage.std() + 1e-7)

            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            value_loss = nn.MSELoss()(values.squeeze(), old_values)

            # Total loss
            loss = actor_loss + 0.5 * value_loss

            # Backpropagate and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Reset buffer after learning
        self.reset_buffers()
