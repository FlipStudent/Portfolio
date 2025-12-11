from typing import List
from building import Building
from DQNagent import Agent
import pandas as pd
from grid import PowerGrid
from environment import Env
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor


def noise_data(data):
    """TODO add noise to the data"""
    return data


def test_loop(agent_1: Agent, agent_2: Agent, temp_data, plot=False):
    temp = pd.read_csv("./data/export.csv")['outside_temperature']
    env = Env(temp)

    temp_1 = []
    temp_2 = []
    rewards_plot = []
    rewards_plot_2 = []
    consumption_1 = []
    consumption_2 = []
    epoch_lines = []
    prices_plot = []

    state_global = env.reset()
    st1_0 = agent_1.reset() + state_global  # extend global with local observation
    st2_0 = agent_2.reset() + state_global  # extend global with local observation

    epoch_lines.append(len(temp_data))

    for timestep, temp_outside in enumerate(temp_data):
        # take action
        action_1, action_1_kwh = agent_1.take_action(np.asarray(st1_0), epsilon=0.0)
        action_2, action_2_kwh = agent_2.take_action(np.asarray(st2_0), epsilon=0.0)

        # update global and local environments (basically equal to env.step, however not all info is shared (building stuff))
        joint_action = (action_1_kwh, action_2_kwh)

        state_global = env.step(joint_action)
        agent_1.building.temperature_difference(action_1_kwh, temp_outside)  # initiate building heating with steps of 2 kwh
        agent_2.building.temperature_difference(action_2_kwh, temp_outside)

        # get current state (env.step()) normally returns new state
        st1_1 = agent_1.get_local_observation() + state_global  # extend global with local observation st+1
        st2_1 = agent_2.get_local_observation() + state_global  # extend global with local observation st+1

        # calculate rewards
        reward_1 = agent_1.get_reward(env)
        reward_2 = agent_2.get_reward(env)

        # new state becomes old state
        st1_0 = st1_1
        st2_0 = st2_1

        # plotting
        temp_1.append(agent_1.building.temp.item())
        temp_2.append(agent_2.building.temp.item())
        rewards_plot.append(reward_1)
        rewards_plot_2.append(reward_2)
        consumption_1.append(agent_1.building.consumption.item())
        consumption_2.append(agent_2.building.consumption.item())

        prices_plot.append(env.grid.current_price)
    if plot:
        plt.clf()
        plt.ylim(bottom=-5, top=40)
        plt.plot(temp_1, label="temperature building 1")
        plt.plot(temp_2, label="temperature building 2")
        plt.plot(temp_data, label="Outside", color="grey")
        plt.axhline(y=19, color='red', linestyle='--')
        plt.axhline(y=21, color='red', linestyle='--')
        # plt.vlines(epoch_lines, ymin=min(min(temp_1), min(temp_2)), ymax=max(max(temp_1), max(temp_2)),
        #            color='green', label="Epoch")
        plt.legend()
        plt.savefig("results/temperatures.png")

        plt.clf()
        plt.plot(prices_plot, label="dynamic price")
        plt.legend()
        plt.savefig("results/prices.png")

        plt.clf()
        plt.plot(rewards_plot, label="rewards_1")
        plt.plot(rewards_plot_2, label="rewards_2")
        plt.legend()
        plt.savefig("results/reward.png")

        plt.clf()
        plt.plot(consumption_1, label="consumption_1")
        plt.plot(consumption_2, label="consumption_2")
        plt.legend()
        plt.savefig("results/consumption.png")

    return np.mean(rewards_plot), np.mean(rewards_plot_2), np.mean(consumption_1), np.mean(consumption_2), np.mean(prices_plot), temp_1, temp_2


def train_loop(n_epochs=500, plot=True):
    building_1 = Building(1)
    building_2 = Building(2)
    agent_1 = Agent(idx=1, building=building_1)
    agent_2 = Agent(idx=2, building=building_2)

    mean_rewards_1 = []
    mean_rewards_2 = []

    mean_consumption_1 = []
    mean_consumption_2 = []

    mean_penalty = []
    mean_price = []

    mean_temperatures_1 = [[] for _ in range(5)]  # collect 5 most recent temperature graphs
    mean_temperatures_2 = [[] for _ in range(5)]


    eps = 1.0
    min_eps = 0.01
    eps_decay = 0.99998

    temp = pd.read_csv("./data/export.csv")['outside_temperature']
    env = Env(temp)

    prices_plot = []
    for epoch in range(n_epochs):  # times to run over the data
        print(f"epoch {epoch}/{n_epochs}, epsilon:{eps}")

        temp_data = noise_data(env.outside_temps)

        state_global = env.reset()
        st1_0 = agent_1.reset() + state_global  # extend global with local observation
        st2_0 = agent_2.reset() + state_global  # extend global with local observation

        for timestep, temp_outside in enumerate(temp_data):
            # take action
            action_1, action_1_kwh = agent_1.take_action(np.asarray(st1_0), eps)
            action_2, action_2_kwh = agent_2.take_action(np.asarray(st2_0), eps)
            eps = max(eps_decay * eps, min_eps)

            # update global and local environments (basically equal to env.step, however not all info is shared (building stuff))
            joint_action = (action_1_kwh, action_2_kwh)

            state_global = env.step(joint_action)
            building_1.temperature_difference(action_1_kwh, temp_outside)  # initiate building heating with steps of 2 kwh
            building_2.temperature_difference(action_2, temp_outside)

            # get current state (env.step()) normally returns new state
            st1_1 = agent_1.get_local_observation() + state_global  # extend global with local observation st+1
            st2_1 = agent_2.get_local_observation() + state_global  # extend global with local observation st+1

            # calculate rewards
            reward_1 = agent_1.get_reward(env)
            reward_2 = agent_2.get_reward(env)
            # print(building_1.temp, reward_1)

            # store buffer
            agent_1.store_transition(st1_0, action_1, reward_1, st1_1)
            agent_2.store_transition(st2_0, action_2, reward_2, st2_1)

            # new state becomes old state
            st1_0 = st1_1
            st2_0 = st2_1

            prices_plot.append(env.grid.current_price)

            if timestep % 50 == 0:
                agent_1.learn()  # will learn with buffers saved in agent class
                agent_2.learn()  # will learn with buffers saved in agent class

        #testloop
        mean_reward_1, mean_reward_2, cons_1, cons_2, price, temp1, temp2 = test_loop(agent_1, agent_2, temp_data, plot=plot)
        mean_rewards_1.append(mean_reward_1)
        mean_rewards_2.append(mean_reward_2)

        mean_consumption_1.append(cons_1)
        mean_consumption_2.append(cons_2)

        mean_price.append(price)
        mean_temperatures_1[epoch%5] = temp1
        mean_temperatures_2[epoch%5] = temp2


    results = {
        "reward 1": mean_rewards_1,
        "reward 2": mean_rewards_2,
        "consumption 1": mean_consumption_1,
        "consumption 2": mean_consumption_2,
        "price": mean_price,
        "temperatures 1": np.mean(np.array(mean_temperatures_1), axis=0),  # average last 5 temperatures of this run
        "temperatures 2": np.mean(np.array(mean_temperatures_2), axis=0),  # average last 5 temperatures of this run

    }

    return results

def main():
    train_loop()


if __name__ == "__main__":
    main()
