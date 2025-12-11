from abstract_building import AbstractBuilding


class Building(AbstractBuilding):
    def __init__(self, heating_mass_constant = 1.08154547e+02, cooling_constant = 8.34797389e-03):
        super().__init__()
        self.heating_mass_constant = heating_mass_constant
        self.cooling_constant = cooling_constant

    def temperature_difference(self, consumption, outside_temp):
        """current proposed model for building temperature
           :param consumption, energy consumpption of the building in kWh,
           :param outside_temp temperature outside,
        """
        self.consumption = consumption
        self.temp += (consumption / self.heating_mass_constant) - self.cooling_constant * (self.temp - outside_temp)
