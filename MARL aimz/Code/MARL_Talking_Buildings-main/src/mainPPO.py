import os
import sys
from typing import List

from building import Building
from PPOagent import PPOAgent
import pandas as pd
from environment import Env
import numpy as np
import matplotlib.pyplot as plt

from datasetAgent import DatasetAgent
from dataset_building import DatasetBuilding
from agent import Agent



def noise_data(temperatures, noise_level=0.5):
    # return temperatures
    noise = np.random.normal(0, noise_level, len(temperatures))

    # Add noise to the original temperatures
    noisy_temperatures = temperatures + noise

    return noisy_temperatures


def train_loop(agents: List[Agent], dataset, save_dir, n_epochs=10, plot=True, plot_frequency_epochs=2):
    temperatures = dataset['outside_temperature']
    env = Env(temperatures)

    n_agents = len(agents)
    env.grid.set_load(n_agents = n_agents)  #set load to 25 * n_agents

    n_timesteps = len(temperatures)

    mat_dimensions = (n_epochs, n_timesteps, n_agents)
    #data:
    rewards_mat = np.empty(mat_dimensions)
    consumptions_mat = np.empty(mat_dimensions)
    penalties_mat = np.empty(mat_dimensions)
    spendings_mat = np.empty(mat_dimensions)
    price_mat = np.empty((n_epochs, n_timesteps))
    temperatures_per_epoch_mat = np.empty((n_timesteps, n_agents))

    actions = [0] * n_agents
    actions_kwh = [0] * n_agents
    log_probs = [0.0] * n_agents
    values = [0] * n_agents
    rewards = [0] * n_agents
    next_states = [[]] * n_agents
    states = [[]] * n_agents
    inside_temps = [0.0] * n_agents
    spendings = [0.0] * n_agents

    n_steps = 2024

    for epoch in range(n_epochs):  # times to run over the data
        if epoch % 100 == 0:
            print(f"epoch {epoch}/{n_epochs}")
        temp_data = noise_data(env.outside_temps)

        state_global = env.reset()  # t=0
        for idx in range(n_agents):
            states[idx] = agents[idx].reset() + state_global

        for timestep, temp_outside in enumerate(temp_data):
            # take action
            for idx in range(n_agents):
                inside_temps[idx] = agents[idx].building.temp
                action, action_kwh, log_prob, value = agents[idx].take_action(np.asarray(states[idx]))   #data returns dummy items except kwh

                actions[idx] = action
                actions_kwh[idx] = action_kwh
                log_probs[idx] = log_prob
                values[idx] = value
                spendings[idx] = env.grid.current_price * action_kwh

            # update global and local environments (basically equal to env.step, however not all info is shared (building stuff))
            joint_action = actions_kwh

            state_global = env.step(joint_action)  # here t increments by 1
            for idx in range(n_agents):
                agents[idx].temperature_difference(actions_kwh[idx], temp_outside)
                next_states[idx] = agents[idx].get_local_observation() + state_global
                rewards[idx] = agents[idx].get_reward(env)
                agents[idx].store_transition(states[idx], actions[idx], rewards[idx], log_probs[idx], values[idx])
                states[idx] = next_states[idx]

                if timestep % n_steps == 0:
                    agents[idx].learn()

            rewards_mat[epoch][timestep] = np.array(rewards)
            price_mat[epoch][timestep] = np.array(timestep)
            consumptions_mat[epoch][timestep] = np.array(np.array(actions_kwh))
            temperatures_per_epoch_mat[timestep] = np.array(inside_temps)
            price_mat[epoch][timestep] = env.grid.current_price
            spendings_mat[epoch][timestep] = np.array(spendings)


        if plot and epoch % plot_frequency_epochs == 0:
            # mean rewards
            plt.clf()
            clipped_rewards_mat = rewards_mat[:epoch]
            mean_rewards = np.mean(clipped_rewards_mat, axis=1)
            mean_rewards = np.transpose(mean_rewards, (1, 0))  # -> (agents, timesteps)
            for idx in range(n_agents):
                plt.plot(mean_rewards[idx], label=f"Mean rewards {agents[idx].name}")

            # plt.ylim((-10,0))
            plt.xlabel("Epoch")
            plt.ylabel("Reward")
            plt.legend()
            plt.savefig(f"./{save_dir}/mean_reward.png")

            # mean spendings
            plt.clf()
            clipped_spendings_mat = spendings_mat[:epoch]
            mean_spendings = np.mean(clipped_spendings_mat, axis=1)
            mean_spendings = np.transpose(mean_spendings, (1, 0))  # -> (agents, timesteps)
            for idx in range(n_agents):
                plt.plot(mean_spendings[idx], label=f"Mean spendings {agents[idx].name}")

            # plt.ylim((-10,0))
            plt.xlabel("Epoch")
            plt.ylabel("Mean spendings ($)")
            plt.legend()
            plt.savefig(f"./{save_dir}/mean_spendings.png")

            # temperatures
            plt.clf()
            temp_transposed = np.transpose(temperatures_per_epoch_mat, (1, 0))  # -> (agents, timesteps)
            for idx in range(n_agents):
                plt.plot(temp_transposed[idx], label=f"Temperatures {agents[idx].name}")
            plt.plot(temp_data, label="Outside", color="grey")
            plt.axhline(y=21, color='red', linestyle='--', alpha=0.5)
            plt.axhline(y=18, color='red', linestyle='--', alpha=0.5)
            plt.ylim(bottom=-5, top=40)
            plt.xlabel("Epoch")
            plt.ylabel("Temperature (C)")
            plt.legend()
            plt.savefig(f"./{save_dir}/temperatures.png")

    # mat_dimensions = (n_epochs, n_timesteps, n_agents)
    mean_rewards = np.mean(rewards_mat, axis=1)
    mean_spendings = np.mean(spendings_mat, axis=1)
    mean_consumptions = np.mean(consumptions_mat, axis=1)
    last_consumption = np.transpose(consumptions_mat, (0, 2, 1))[-1]

    result_dictionary = {}
    for idx, agent in enumerate(agents):
        result_dictionary[agent.name] = {}
        result_dictionary[agent.name]['rewards'] = np.transpose(mean_rewards, (1,0))[idx]
        result_dictionary[agent.name]['spendings'] = np.transpose(mean_spendings, (1,0))[idx]
        result_dictionary[agent.name]['mean_consumption'] = np.transpose(mean_consumptions, (1,0))[idx]
        result_dictionary[agent.name]['last_consumption'] = last_consumption[idx]
        result_dictionary[agent.name]['last_temp'] = np.transpose(temperatures_per_epoch_mat, (1,0))[idx]

    return result_dictionary


def run_experiment(baseline, dataset, save_dir, n_epochs=1000, plot=True):
    save_dir = f"{save_dir}/intermediate"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    agents = []
    if baseline:
        agents.append(PPOAgent(name=f"ppo_baseline", building=Building(), includePriceInReward=False))
        agents.append(DatasetAgent(name="dataset", building=DatasetBuilding(dataset), dataset=dataset))
    else:
        num_ppo_price_agents = 2
        for i in range(num_ppo_price_agents):
            agent = PPOAgent(name=f"ppo_{i}", building=Building())
            agents.append(agent)

    return train_loop(agents, dataset, save_dir=save_dir, n_epochs=n_epochs, plot=plot)


if __name__ == "__main__":
    dataset = pd.read_csv("./data/export.csv")
    save_dir = "results/mainPPOrun"
    run_experiment(False, dataset, save_dir)
