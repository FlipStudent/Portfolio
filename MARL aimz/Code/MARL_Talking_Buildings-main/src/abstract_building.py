from abc import abstractmethod


class AbstractBuilding:
    def __init__(self):
        self.temp = 18
        self.consumption = 0

        self.wish_temp = 20
        self.office_hours = [True if 7 < i < 18 else False for i in range(24)]
        pass

    @abstractmethod
    def temperature_difference(self, consumption, outside_temp):
        pass
