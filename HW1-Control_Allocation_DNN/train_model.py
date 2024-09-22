import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from models import *

# Load a data point
data = np.load('data/data_2.npy')
print(f"Data loaded")

# Split data into test/train u and tau
split_index = int(.80 * len(data[0,:]))
train_u = torch.tensor(data[:5,:split_index], dtype=torch.float32).T
train_tau = torch.tensor(data[5:,:split_index], dtype=torch.float32).T
test_u = torch.tensor(data[:5,split_index:], dtype=torch.float32).T
test_tau = torch.tensor(data[5:,split_index:], dtype=torch.float32).T

# plt.figure()
# plt.plot(train_tau[:,0], '.', label='train')
# plt.plot(test_tau[:,0], '.', label='test')
# plt.show()

# Calculate norms based on training data
train_tau_mean = train_tau.mean(dim=0)
train_tau_std = train_tau.std(dim=0)

# Scale train and test data by train mean/std
train_tau_norm = (train_tau - train_tau_mean) / train_tau_std
test_tau_norm = (train_tau - train_tau_mean) / train_tau_std

# Load in train and test tensors to dataset
train_dataset = TensorDataset(train_tau_norm, train_tau_norm)
test_dataset = TensorDataset(test_tau_norm, test_tau_norm)

# Hyper Params
batch_size = 5000
learning_rate = 0.001
num_epochs = 1000
l2_lambda = 0 #0.01
k = [10**0, 10**0, 10**-1, 10**-7, 10**-7, 10**-1]
lengths = [-14, 14.5, -2.7, 2.7]
u_max = torch.tensor([30000, 30000, 30000, 180, 180])
u_diff_max = torch.tensor([1000, 1000, 1000, 10, 10])
azimuth_limits = torch.tensor([[-100, -80], [80, 100]])

print(f"Batch Size: {batch_size}, Learning Rate: {learning_rate}, L2 Lambda: {l2_lambda}")

# Load in data to data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = allocationNN().to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define Loss
def calcLoss(batch_data, u, tau_rec, u_max, u_diff_max, azimuth_limits, k, lengths):
    a2_rad = torch.deg2rad(u[:, 3])
    a3_rad = torch.deg2rad(u[:, 4])
    t1 = torch.cos(a2_rad) * u[:, 1] + torch.cos(a3_rad) * u[:, 2]
    t2 = u[:, 0] + torch.sin(a2_rad) * u[:, 1] + torch.sin(a3_rad) * u[:, 2]
    t3 = (lengths[1] * u[:, 0] + 
        (lengths[0] * torch.sin(a2_rad) - lengths[2] * torch.cos(a2_rad)) * u[:, 1] + 
        (lengths[0] * torch.sin(a3_rad) - lengths[3] * torch.cos(a3_rad)) * u[:, 2])
    y_cmd = torch.stack([t1, t2, t3], dim=1)

    L0 = torch.mean((batch_data - y_cmd)**2)
    
    # L1 = MSE(y-y_rec)
    L1 = torch.mean((batch_data - tau_rec)**2)

    # L2 = sum(max(|u| - u_lim, 0))
    L2 = torch.sum(torch.max(torch.abs(u) - u_max.to(device), torch.tensor(0.0, device=device)))

    # L3 = sum(max|u - u_prev| - du_lim, 0)
    u_diff = torch.abs(torch.diff(u, dim=0))
    L3 = torch.sum(torch.max(u_diff - u_diff_max.to(device), torch.tensor(0.0, device=device)))

    # L4 = |u0| ^ 3/2 + |u1| ^ 3/2 + |u2| ^ 3/2
    L4 = torch.sum(torch.pow(torch.abs(u[:, :3]), 3/2))

    # L5 = sum((𝒖̂ 𝑙 < 𝛼 1,𝑐 ) × (𝒖̂ 𝑙 > 𝛼 0,𝑐 )) + sum((𝒖̂ 𝑙 < 𝛼 1,𝑐 ) × (𝒖̂ 𝑙 > 𝛼 0,𝑐 ))
    a2 = u[:,3]
    a3 = u[:,4]
    L5 = torch.sum(torch.logical_and(a2 < azimuth_limits[0][1], a2 > azimuth_limits[0][0]))
    L5 += torch.sum(torch.logical_and(a3 < azimuth_limits[1][1], a3 > azimuth_limits[1][0]))

    # Total Loss
    return k[0] * L0 + k[1] * L1 + k[2] * L2 + k[3] * L3 + k[4] * L4 + k[5] * L5

def l2_regularization(model):
    l2_loss = 0.0
    for param in model.parameters():
        l2_loss += torch.norm(param, 2)
    
    return l2_loss

# Train model
epochs = []
train_losses = []
test_losses = []
try:
    for epoch in range(num_epochs):

        # Train Model
        model.train()
        epoch_loss = 0.0

        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            u, tau_rec = model(batch_data)

            total_loss = calcLoss(batch_data, u, tau_rec, u_max, u_diff_max, azimuth_limits, k, lengths)
            total_loss += l2_lambda * l2_regularization(model)

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)

        epochs.append(epoch)
        train_losses.append(avg_epoch_loss)

        # Evaluate model
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

                u, tau_rec = model(batch_data)
                total_loss = calcLoss(batch_data, u, tau_rec, u_max, u_diff_max, azimuth_limits, k, lengths)
                test_loss += total_loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)


        print(f"Epoch: {epoch}/{num_epochs}, Train Loss: {avg_epoch_loss:.5f}, Test Loss: {avg_test_loss:.5f}")
except KeyboardInterrupt:
    print("Training Interrupted")


plt.figure()
plt.plot(epochs, train_losses, '.', label='Training Loss')
plt.plot(epochs[:len(test_losses)], test_losses, '.', label='Testing Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epoch\n" + f"Batch Size: {batch_size}, Learning Rate: {learning_rate}, L2 Lambda: {l2_lambda}")
plt.legend()
plt.yscale('log')
plt.ylim([min([min(train_losses), min(test_losses)]), max([max(train_losses), max(test_losses)])])
plt.grid()
plt.show()