from abc import abstractmethod
from typing import List
from environment import Env
from building import Building
import numpy as np
import torch

from abstract_building import AbstractBuilding


class Agent:
    def __init__(self, name, building: AbstractBuilding):
        self.name = name
        self.input_len = 5
        self.action_dim = 15  # Discrete action space with 10 possible actions
        self.kWh_step = 4

        self.building = building

        self.gamma = 0.99
        self.lamda = 0.95
        self.clip_param = 0.2
        self.memory_size = 2024  # nsteps
        self.nsteps = 2024
        self.batch_size = 256
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"This agent is running on: {self.device}")

        self.state_buffer = np.zeros((self.memory_size, self.input_len), dtype=np.float32)
        self.new_state_buffer = np.zeros((self.memory_size, self.input_len), dtype=np.float32)
        self.action_buffer = np.zeros(self.memory_size, dtype=np.int64)  # Storing discrete actions as int
        self.reward_buffer = np.zeros(self.memory_size, dtype=np.float32)
        self.log_prob_buffer = np.zeros(self.memory_size, dtype=np.float32)
        self.value_buffer = np.zeros(self.memory_size, dtype=np.float32)
        self.advantage_buffer = np.zeros(self.memory_size, dtype=np.float32)
        self.memory_counter = 0
        self.penalties = []

    def reset_buffers(self):
        self.state_buffer = np.zeros((self.memory_size, 5), dtype=np.float32)
        self.new_state_buffer = np.zeros((self.memory_size, 5), dtype=np.float32)
        self.action_buffer = np.zeros(self.memory_size, dtype=np.int64)
        self.reward_buffer = np.zeros(self.memory_size, dtype=np.float32)
        self.log_prob_buffer = np.zeros(self.memory_size, dtype=np.float32)
        self.value_buffer = np.zeros(self.memory_size, dtype=np.float32)
        self.advantage_buffer = np.zeros(self.memory_size, dtype=np.float32)

    def reset(self) -> List:
        return [self.building.temp, self.building.consumption]

    def get_local_observation(self) -> List:
        return [self.building.temp, self.building.consumption]

    def store_transition(self, state, action, reward, log_prob, value):
        index = self.memory_counter % self.memory_size
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.log_prob_buffer[index] = log_prob
        self.value_buffer[index] = value
        self.memory_counter += 1

    def temperature_difference(self, consumption, outside_temp):
        self.building.temperature_difference(consumption, outside_temp)

    @abstractmethod
    def get_reward(self, env: Env):
        pass

    @abstractmethod
    def take_action(self, observation: np.ndarray):
        pass

    @abstractmethod
    def learn(self):
        pass
