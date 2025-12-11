from abstract_building import AbstractBuilding

class DatasetBuilding(AbstractBuilding):
    def __init__(self, dataset):
        super().__init__()
        self.temperatures = dataset['room_temperature']
        self.max_tsteps = len(self.temperatures)
        self.t = 0
        self.temp = self.temperatures[0]

    def temperature_difference(self, consumption, outside_temp):
        """current proposed model for building temperature
           :param consumption, energy consumpption of the building in kWh,
           :param outside_temp temperature outside,
        """

        self.temp = self.temperatures[self.t%self.max_tsteps]
        self.t += 1