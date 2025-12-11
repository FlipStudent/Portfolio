import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import random_split 
from sklearn.model_selection import train_test_split
import random

def plot_random_days(inputs, outputs, targets):
    indices = random.sample(range(len(outputs)), 4)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    for n, idx in enumerate(indices):
        i = int(n / 2)
        j = int(n % 2)
        # print(f"[{i}][{j}] - {idx} ({n})")
        ax[i][j].plot(outputs[idx], color='grey')
        ax[i][j].plot(targets[idx], color='black')
        ax[i][j].plot(inputs[idx][:24], color='blue')
        ax[i][j].plot(inputs[idx][24:], color='orange')
    pass

def plot_running_forecast(inputs, model, targets):
    
    # for hour in range()
    pass



# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size=24*3, hidden_size=64, output_size=1):
        super(MLP, self).__init__()
        # Define the layers of the network
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Hidden layer to another hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size) # Hidden layer to output layer
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        # Forward pass
        x = self.relu(self.fc1(x))  # Pass input through first layer and activation
        x = self.relu(self.fc2(x))  # Pass through second layer and activation
        x = self.fc3(x)             # Output layer (no activation for regression)
        return x

data = pd.read_csv('./data/export.csv')

usage = torch.from_numpy(data['energy_meter_fixed'].values)
temp_o = torch.from_numpy(data['outside_temperature'].values)
temp_i = torch.from_numpy(data['room_temperature'].values)

window_size = 24
n_inputs_vars = 3

# Initialize the MLP
input_size = n_inputs_vars * window_size
hidden_size = 64
output_size = 1

model = MLP(input_size, hidden_size, output_size)

# Define a loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_days = int(len(usage)/24)
# inputs = torch.empty(size=(n_days, 24*2))
# targets = torch.empty(size=(n_days, 24))

inputs = []
targets = []

# weekend_inputs = []
# weekend_targets = []

# # Input row represents: [outside temperature (24), inside temperature (24), energy usage (24)]
# # Target val represents: [inside temperature t+1 (1)]
for hour in range(len(usage)-25):
    index_hours = range(hour, hour+24)
    inputs.append(torch.cat([temp_o[index_hours], temp_i[index_hours], usage[index_hours]], dim = 0))
    targets.append(temp_i[hour+25])

inputs = torch.stack(inputs, dim=0).to(torch.float32)
targets = torch.tensor(targets).to(torch.float32)

# # Input row represents: [outside temperature (24), energy usage (24)]
# # Target row represents: [inside temperature (24)]
# daily_hours = torch.tensor(range(0,24))
# for day in range(n_days):
#     start = 24*day
#     index_hours = range(start, start + 24)
#     if torch.mean(usage[index_hours]) < 8:
#         weekend_inputs.append(torch.cat([temp_o[index_hours], usage[index_hours], daily_hours], dim = 0))
#         weekend_targets.append(temp_i[start:start+24])
#         continue
#     inputs.append(torch.cat([temp_o[index_hours], usage[index_hours]], dim = 0))
#     targets.append(temp_i[start:start+24])
#     # inputs[day] = torch.cat([temp_o[index_hours], usage[index_hours]], dim = 0)
#     # targets[day] = temp_i[start:start+24]

# inputs = torch.stack(inputs, dim=0).to(torch.float32)
# targets = torch.stack(targets, dim=0).to(torch.float32)

# weekend_inputs = torch.stack(weekend_inputs, dim=0).to(torch.float32)
# weekend_targets = torch.stack(weekend_targets, dim=0).to(torch.float32)

X_train, X_test, y_train, y_test = train_test_split(inputs,
                                                    targets,
                                                    test_size=0.2)

# Forward pass
num_epochs = 10
for epoch in range(num_epochs):
    for day in range(len(X_train)):
        output = model(X_train[day])
        target = y_train[day]
        # Calculate the loss
        loss = criterion(output, target)

        # Backward pass and optimization step
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()        # Backpropagate the loss
        optimizer.step()       # Update the model's parameters
    print(f"Ep({epoch}): L={loss.item()}")

with torch.no_grad():
    test_outputs = model(X_test).squeeze()
    test_targets = y_test
    loss = criterion(test_outputs, test_targets)
    # for day in range(len(X_test)):
    #     output = model(X_test[day])
    #     target = y_test[day]
    #     # Calculate the loss
    #     loss = criterion(output, target)

    # print("Output:", outputs)
    print("Loss:", loss.item())

    plot_random_days(inputs = X_test, outputs=test_outputs, targets=test_targets)
    forecast_indices = range(0, 200)
    plot_running_forecast(inputs = X_test[forecast_indices], model=model, targets = y_test[forecast_indices])
    pass