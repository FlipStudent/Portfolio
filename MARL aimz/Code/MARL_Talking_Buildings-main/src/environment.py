from typing import List

from grid import PowerGrid
from building import Building
import numpy as np


class Env:
    """Method to simulate a sort of gym environment,
    this doesnt work fully as we ofcourse work in a marl environment, so some environmental parameters are private.
    These private environmental parameters are found in the PPOagent.py:get_local_observation or reset methods"""

    def __init__(self, temp_list):
        self.outside_temps = temp_list
        self.grid = PowerGrid()
        self.current_hour = 0
        self.t = 0

    def reset(self) -> List:
        self.t = 0
        self.grid = PowerGrid()

        return [0, self.outside_temps[self.t], self.grid.price_standard]

    def step(self, joint_action) -> List:
        """ :param joint_action joint action of the agents
            :returns the global observations, time, temperature, price"""
        temperature = self.outside_temps[self.t]
        hour = self.t % 24
        hour_float = hour / 24
        price = self.grid.update(joint_action)

        self.t += 1
        return [hour_float, temperature, price]


    def get_time(self):
        return self.t % 24
