import os
import sys
import pandas as pd
from mainPPO import run_experiment
import numpy as np
import matplotlib.pyplot as plt
import pickle

SAVE_DIR = "results"


class Plotting:
    def __init__(self, name:str, final_save_dir: str):
        self.all_results = {}
        self.agent_names = None
        self.data_names = None
        self.name = name
        self.runs = 0
        self.final_save_dir = final_save_dir

    def get_names(self):
        self.runs = len(self.all_results.keys())
        self.agent_names = list(self.all_results[0].keys())
        self.data_names = list(self.all_results[0][self.agent_names[0]].keys())

    def add_run(self, data_dict: dict, run_idx):
        """add run dictionary to all data"""
        self.all_results[run_idx] = data_dict
        self.runs += 1
        if not self.agent_names:
            self.agent_names = list(data_dict.keys())
        if not self.data_names:
            self.data_names = list(data_dict[self.agent_names[0]].keys())


    def plot(self, data_key, ylab, title, fig_name, xlab="Epoch", additional_calls=None, upper_limit=False, exclude = []):
        plt.clf()
        colors = ['blue', 'orange', 'red', 'green', 'brown']

        for idx, key in enumerate(self.agent_names):
            if key in exclude:
                continue

            total = []
            for i in range(self.runs):
                total.append(self.all_results[i][key][data_key])

            data_arr = np.asarray(total)
            mean = np.mean(data_arr, axis=0)
            std = np.std(data_arr, axis=0)
            plt.plot(mean)
            x = np.arange(data_arr.shape[1])

            plt.plot(mean, color=colors[idx], label=f'{key}')
            if upper_limit:
                upper = np.minimum(mean + std, 0)
            else:
                upper = mean + std

            plt.fill_between(x, mean - std, upper, color=colors[idx], alpha=0.2)

        if additional_calls:
            for call in additional_calls:
                call()

        plt.legend()
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)

        plt.savefig(f"{self.final_save_dir}/{fig_name}")

    def temp_lines(self):
        plt.ylim(bottom=-5, top=40)
        plt.axhline(y=21, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=18, color='red', linestyle='--', alpha=0.5)

    def plot_all(self, name="ppo"):
        self.plot(data_key="rewards", ylab="Reward", title="Mean Rewards", fig_name=f"rewards_{self.name}.png",
                  upper_limit=True, exclude=["dataset"])
        self.plot(data_key="spendings", ylab="Total cost (€)", title="Mean Price", fig_name=f"price_{self.name}.png")

        self.plot(data_key="mean_consumption", ylab="Consumption (kWh)", title="Mean Consumption",
                  fig_name=f"consumption_{self.name}.png", exclude=["dataset"])

        self.plot(data_key="last_consumption", ylab="Consumption (kWh)",xlab="Time (hour)", title="Consumption",
                  fig_name=f"consumption_1_epoch_{self.name}.png")

        self.plot(data_key="last_temp", ylab="Temperature (Celsius)",xlab="Time (hour)", title="Temperature",
                  fig_name=f"temperatures_{self.name}.png", additional_calls=[self.temp_lines])

    def save(self):
        with open(f'{self.final_save_dir}/final_results_{self.name}.pkl', 'wb') as file:
            pickle.dump(self.all_results, file)

    def load(self):
        with open(f'{self.final_save_dir}/final_results_{self.name}.pkl', 'rb') as file:
            self.all_results = pickle.load(file)
        self.get_names()


def plot_results(save_dir):
    plot_class = Plotting("ppo", save_dir)
    plot_class.load()
    plot_class.plot_all()


def from_save(save_dir):
    plot_class = Plotting("ppo", save_dir)
    plot_class.load()
    plot_class.plot_all()


def generate_results(baseline: bool, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    final_save_dir = f"{save_dir}/final"
    if not os.path.exists(final_save_dir):
        os.makedirs(final_save_dir)

    train_runs = 5
    n_epochs = 1000

    plot_class = Plotting(name="ppo", final_save_dir=final_save_dir)

    dataset = pd.read_csv("./data/export.csv")
    for run_idx in range(train_runs):
        print(f"Run number: {run_idx + 1}")
        result_dict = run_experiment(baseline, dataset, save_dir=save_dir, n_epochs=n_epochs)
        plot_class.add_run(result_dict, run_idx)
        plot_class.save()

    plot_class.plot_all()


if __name__ == '__main__':
    directory_suffix = ""
    if len(sys.argv) < 2:
        print("Usage: mainPPO.py <baseline|experiment> [directory_suffix]")
        exit()
    if len(sys.argv) == 3:
        directory_suffix = sys.argv[2]

    setup = sys.argv[1]
    if setup == "baseline":
        print("Running baseline...")
        generate_results(True, save_dir=f"{SAVE_DIR}/{setup}_{directory_suffix}")
    elif setup == "experiment":
        print("Running experiment...")
        generate_results(False, save_dir=f"{SAVE_DIR}/{setup}_{directory_suffix}")
    else:
        print("Usage: mainPPO.py <baseline|experiment> [directory_suffix]")
        exit()
