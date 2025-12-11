from typing import List
from building import Building
import numpy as np


class PowerGrid:
    def __init__(self, n_agents:int = 4):
        self.max_capacity = n_agents * 25  # kWh idk   #TODO TWEAK
        self.n_agents = n_agents
        self.current = 0  # current70 used

        self.price_standard = 0.15  # cent per kWh
        self.current_price = self.price_standard

    def set_load(self, n_agents:int = 4):
        self.max_capacity = n_agents * 25

    def set_current_price(self):
        price = 0.8 * np.exp((self.current - self.max_capacity)) + 0.005 * self.current + 0.001   #TODO tweak
        self.current_price = min(price, 100)

    def get_demand(self, joint_action):
        # Add demand of all buildings
        total_demand = 0
        for consumption in joint_action:
            total_demand += consumption
        self.current = total_demand

    def update(self, joint_action):
        self.get_demand(joint_action)
        # Set price standard (dependent on time) and grid power price
        self.set_current_price()
        return self.current_price
