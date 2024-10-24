from Utilities import *
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


data = pd.read_csv('data/dataArray_table.csv', nrows=100000)
Data = DataClass(data)

num_epochs = 200
learning_rate = 0.001

model = SimpleRNN(6, 32, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
running_loss = []

for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    Data.reset()
    done = False
    batch = 0
    epoch_loss = 0
    while not done:
        # print(f"\tBatch: {batch}")
        run, done = Data.next()
        optimizer.zero_grad()

        output = model(run.unsqueeze(0)) # Run size: [1, n, 6]
    
        # loss = criterion(output[0, :, :], run) # Is this right???
        input_sequence = run.unsqueeze(0)
        loss = criterion(output, input_sequence)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # plotTrajTensor(output)
        # print(f"\t\tOutput: {output}")

        # print(f"\t\tLast Idx: {Data.current_idx}, shape: {run.shape}")
        batch += 1

    avg_epoch_loss = epoch_loss / batch
    running_loss.append(avg_epoch_loss)
    print(f"Epoch Average Loss: {avg_epoch_loss:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(running_loss)
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.show()