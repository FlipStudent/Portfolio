# dT = usage * 3_600_000 / C - k (temp_i - temp_o)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def temperature_difference(usage, temp_i, temp_o, Cm, k):
    """current proposed model for building temperature
       :param usage, energy consumpption of the building in kWh,
       :param temp_i temperature inside,
       :param temp_o temperature outside,
       :param Cm heating+mass constant combined,
       :param k cooling constant"""
    return (usage / Cm) - k * (temp_i - temp_o)


def fit_func(vars, C, k):
    usage, temp_i, temp_o = vars
    return temperature_difference(usage, temp_i, temp_o, C, k)


def fit_curve(usage, temp_i, temp_o, dT):
    variables = np.vstack([usage, temp_i, temp_o])
    out = curve_fit(f=fit_func, xdata=(usage, temp_i, temp_o), ydata=dT)
    print(out[0])

    temps_dt = dt_to_temps(dT, temp_i[0])
    temps_fit = dt_to_temps(fit_func((usage, temp_i, temp_o), *out[0]), temp_i[0])
    # plt.plot(range(len(dT)), temps_dt , 'b-', label='data')
    plt.plot(range(len(dT)), temps_fit, 'r-', label='Inside temp pred.')
    plt.plot(range(len(dT)), temp_i, 'y-', label='Inside temp from dataset')
    plt.plot(temp_o, label='Outside temp')
    plt.legend()

    plt.show()


def dt_to_temps(dts, start_temp=18):
    new_temps = []
    current = start_temp
    for dt in dts:
        current += dt
        new_temps.append(current)

    return new_temps


if __name__ == '__main__':
    data = pd.read_csv('./data/export.csv')

    usage = data['energy_meter_fixed'].values

    #print(usage)
    temp_i = data['room_temperature'].values
    #print(temp_i)
    temp_o = data['outside_temperature'].values
    #plt.plot(range(len(temp_i)), usage)
    #plt.plot(range(len(temp_i)), temp_i)
    #plt.plot(range(len(temp_i)), temp_o)
    #plt.show()


    #exit()


    dT = [0]
    for idx in range(1, len(usage) ):
        dT.append(temp_i[idx] - temp_i[idx-1])
    fit_curve(usage, temp_i, temp_o, dT)
