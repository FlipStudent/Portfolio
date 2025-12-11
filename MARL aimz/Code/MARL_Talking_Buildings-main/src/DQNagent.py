from typing import List

from building import Building
from network import NeuralNetwork
import torch
import torch.nn as nn
import numpy as np
from environment import Env


class Agent():
    def __init__(self, idx, building: Building):
        self.input_len = 5
        self.output_size = 15
        self.kWh_step = 2

        self.idx = idx
        self.building = building
        self.Q_eval = NeuralNetwork(self.input_len, self.output_size)
        self.Q_target = NeuralNetwork(self.input_len, self.output_size)
        self.update_target()

        self.learn_steps = 0
        self.update_frequency = 50

        self.batch_size = 1024
        self.gamma = 0.995
        self.memory_size = 50000
        self.memory_counter = 0
        self.device = "cuda"

        self.action_space = [i for i in range(5)]
        self.state_buffer = np.zeros((self.memory_size, self.input_len), dtype=np.float32)
        self.new_state_buffer = np.zeros((self.memory_size, self.input_len), dtype=np.float32)
        self.action_buffer = np.zeros(self.memory_size, dtype=np.int64)
        self.reward_buffer = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_buffer = np.zeros(self.memory_size, dtype=np.bool_)

    def reset_buffers(self):
        self.state_buffer = np.zeros((self.memory_size, 6), dtype=np.float32)
        self.new_state_buffer = np.zeros((self.memory_size, 6), dtype=np.float32)
        self.action_buffer = np.zeros(self.memory_size, dtype=np.int64)
        self.reward_buffer = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_buffer = np.zeros(self.memory_size, dtype=np.bool_)

    def reset(self) -> List:
        # self.reset_buffers()
        return [self.building.temp, self.building.consumption]

    def get_local_observation(self) -> List:
        return [self.building.temp, self.building.consumption]

    def take_action(self, observation: np.ndarray, epsilon: float):  # , current_energy_consumption, epsilon):
        if np.random.random() < epsilon:
            output_idx = torch.tensor(np.random.randint(0, self.output_size))
        else:
            tensor = torch.tensor(observation, dtype=torch.float32)
            output = self.Q_eval(tensor)
            output_idx = torch.argmax(output)

        return output_idx, output_idx * self.kWh_step

    def update_target(self):
        """Method to update Q_target to Q_eval"""
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def get_reward(self, env: Env):
        # calculate reward based on time and temperature
        # if(self.building.office_hours[env.current_hour]):
        #     current_temp = self.building.temp
        #     # Check if the temperature is within the desired range (19 to 21)
        #     if current_temp < 19:
        #         temp_penalty = 19 - current_temp  # Penalty for being below the range
        #     elif current_temp > 21:
        #         temp_penalty = current_temp - 21  # Penalty for being above the range
        #     else:
        #         temp_penalty = 0  # No penalty if within range
        #
        #     temp_penalty = -((self.building.wish_temp - current_temp) ** 2)  # Negative reward for being out of range
        # else:
        #     temp_penalty = 0

        # temp_penalty = -((self.building.wish_temp - self.building.temp) ** 2)
        #
        # # calculate the penalty based on price
        # current_grid_price = env.grid.current_price
        # current_consumption = self.building.consumption
        #
        # price = current_grid_price * current_consumption  # TODO check of dit allemaal kWh is!
        # price_penalty = -(0.001*np.exp(price))
        #
        # return temp_penalty # + price_penalty
        # calculate reward based on time and temperature
        current_temp = self.building.temp

        if self.building.office_hours[env.get_time()]:
            temp_penalty = -((self.building.wish_temp - self.building.temp) ** 2)
        else:
            temp_penalty = 0

        current_grid_price = env.grid.current_price
        current_consumption = self.building.consumption
        price = current_grid_price * current_consumption
        price_penalty = -(0.001 * np.exp(price))

        return temp_penalty + price_penalty

    def update_buffer(self, buffer, value):
        index = self.memory_counter % self.memory_size
        buffer[index] = value

    def store_transition(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            new_state: np.ndarray,
            # done: bool
    ) -> None:

        self.update_buffer(self.state_buffer, state)
        self.update_buffer(self.new_state_buffer, new_state)
        self.update_buffer(self.action_buffer, action)
        self.update_buffer(self.reward_buffer, reward)
        # self.update_buffer(self.terminal_buffer, done)

        # update memory counter
        self.memory_counter += 1

    def learn(self):
        if self.memory_counter < self.batch_size:
            return
        self.learn_steps += 1

        max_mem = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)  # select random a batch (indices)

        # get batches of data
        state_batch = torch.tensor(self.state_buffer[batch]).to(self.device)
        action_batch = torch.tensor(self.action_buffer[batch]).to(self.device)
        reward_batch = torch.tensor(self.reward_buffer[batch]).to(self.device)
        new_state_batch = torch.tensor(self.new_state_buffer[batch]).to(self.device)

        # calculate Q values (Q_t, a)
        state_action_values = self.Q_eval.forward(state_batch).gather(dim=1, index=action_batch.unsqueeze(
            1)).squeeze()  # unsqueeze to use indices, squeeze to 2d tensor -> 1d tensor

        # print(torch.max(torch.abs(state_action_values)).item())
        # calculate Q values (Q_t+1) using target network
        with torch.no_grad():
            next_state_values = self.Q_target.forward(new_state_batch).max(dim=1)[0]  # 0 for actions, (1 == indices)
            # calculate the TD target (r + y*max(Q_t+1))
            expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        # calculate loss between TD target and Q_t
        loss = self.Q_eval.loss(state_action_values, expected_state_action_values)

        # update
        self.Q_eval.optimizer.zero_grad()  # reset gradiants to 0
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_eval.parameters(), max_norm=1.0)  # clip gradients
        self.Q_eval.optimizer.step()

        # update target network
        if self.learn_steps % self.update_frequency == 0:
            self.update_target()
