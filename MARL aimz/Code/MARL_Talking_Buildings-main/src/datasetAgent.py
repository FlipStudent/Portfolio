from building import Building
import numpy as np
from agent import Agent
from environment import Env
from abstract_building import AbstractBuilding


class DatasetAgent(Agent):
    def __init__(self, name, building: AbstractBuilding, dataset):
        super(DatasetAgent, self).__init__(name, building)
        self.dataset = dataset
        self.consumption_data = self.dataset["energy_meter_fixed"]
        self.data_length = len(self.consumption_data)
        self.t = 0

    def get_reward(self, env: Env):
        return 0

    def take_action(self, observation: np.ndarray):
        consumption = self.consumption_data[self.t % self.data_length]
        self.t += 1
        return 0, consumption, 0, 0

    def learn(self):
        pass
