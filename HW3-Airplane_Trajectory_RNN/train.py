from Utilities import *
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
import time
print("Script is running from:", script_dir)
file_path = os.path.join(script_dir, 'data_array_smaller.csv')
data = pd.read_csv(file_path, nrows=100000)
#print(data.head())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Data = DataClass(data)
hyperParams = {
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 0.0001,
    'weight_decay': 0.0001,
    'input_size': 7,
    'hidden_size': 32,

}
num_epochs = hyperParams['num_epochs']
learning_rate = hyperParams['learning_rate']
model = SimpleRNN(hyperParams['input_size'], hyperParams['hidden_size'], 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
running_loss = []
start_time = time.time()
for epoch in range(num_epochs):
    epoch_start_time = time.time()  # Start time for the current epoch
    print(f"Epoch: {epoch}")
    Data.reset()
    done = False
    batch = 0
    epoch_loss = 0
    while not done:
        run, done = Data.next()

        # print(f"\tBatch: {batch}")
        for point_num in range(len(run)):
            single_point = run[point_num].unsqueeze(0)  # Unsqueeze to add batch dimension
            #print(single_point)
            single_point = single_point.to(device)
            optimizer.zero_grad()
            #print(run)
            output = model(run.unsqueeze(0)) # Run size: [1, n, 6]
        
            # loss = criterion(output[0, :, :], run) # Is this right???
            input_sequence = run.unsqueeze(0)
            target_sequence = input_sequence[:, -1, :]  # Assuming you want the last time step
            loss = criterion(output, target_sequence)

            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        # plotTrajTensor(output)
        # print(f"\t\tOutput: {output}")

        # print(f"\t\tLast Idx: {Data.current_idx}, shape: {run.shape}")
        batch += 1
    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch {epoch} took {epoch_duration:.2f} seconds")
    avg_epoch_loss = epoch_loss / batch
    running_loss.append(avg_epoch_loss)
    print(f"Epoch Average Loss: {avg_epoch_loss:.6f}")
total_duration = time.time() - start_time
print(f"Total training time: {total_duration:.2f} seconds")
script_dir = os.path.dirname(os.path.abspath(__file__))
#model_name = input('Enter name to save model, enter otherwise: ')
model_name = 'overnight_run'
model_dir = os.path.join(script_dir, 'models')
if model_name:
    if not os.path.exists(model_dir):
        print(f"'models' directory not found. Creating {model_dir}")
        os.makedirs(model_dir)

    torch.save(model, 'models/' + model_name + '.pth')
    print(f"Model saved in models/{model_name}.pth")
plt.figure(figsize=(10, 5))
plt.plot(running_loss)
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.show()