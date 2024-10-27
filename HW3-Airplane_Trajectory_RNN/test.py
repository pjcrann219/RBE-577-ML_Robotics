import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from DataClass import DataClass
from rnnClass import RNN

# Load data
data = pd.read_csv('data/dataArray_table.csv', nrows=250000)
Data = DataClass(data)

# Initialize model with same parameters as training
input_size = 4
hidden_size = 32
output_size = 3
model = RNN(input_size, hidden_size, output_size, Data.max_seq_length)

# Load the saved model
checkpoint = torch.load('models/CurrentModel.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode

# Run inference on a few examples
n_examples = 10

plt.figure(figsize=(15, 5))
for i in range(n_examples):
    # Get a sample trajectory
    input, true_traj, seq_length = Data.get_run_torch(np.random.randint(low=0, high=max(data['group_id'])))
    
    # Run model inference
    with torch.no_grad():
        input_batch = input.unsqueeze(0)
        predicted_traj = model(input_batch, seq_length)
        predicted_traj = predicted_traj.squeeze(0)
    
    # Plot only up to the actual sequence length
    true_traj = true_traj[:seq_length]
    predicted_traj = predicted_traj[:seq_length]
    
    plt.subplot(2, n_examples//2, i+1)

    plt.plot(true_traj[:, 0], true_traj[:, 1], 'b-', label='True')
    plt.plot(predicted_traj[:, 0].detach(), predicted_traj[:, 1].detach(), 'r--', label='Predicted')
    
    plt.title(f'Run {i+1}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Print some statistics
for i in range(n_examples):
    input, true_traj, seq_length = Data.get_run_torch(i)
    with torch.no_grad():
        input_batch = input.unsqueeze(0)
        predicted_traj = model(input_batch, seq_length)
        predicted_traj = predicted_traj.squeeze(0)[:seq_length]  # Only take valid sequence length
        
    true_traj = true_traj[:seq_length]
    
    mse = nn.MSELoss()(predicted_traj, true_traj)
    print(f"\nTrajectory {i+1}:")
    print(f"MSE: {mse.item():.6f}")
    print(f"Input features: {input.numpy()}")
    print(f"Sequence length: {seq_length}")