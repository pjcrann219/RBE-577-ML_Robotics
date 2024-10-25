import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from DataClass import DataClass
from rnnClass import RNN

data = pd.read_csv('data/dataArray_table.csv', nrows=2500)
Data = DataClass(data)

input_size = 4
hidden_size = 32
output_size = 3
learning_rate = 0.00005
num_epochs = 300

model = RNN(input_size, hidden_size, output_size, Data.max_seq_length)
criterion = nn.MSELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

running_loss = []
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}", end='')
    Data.reset()
    done = False
    batch = 0
    epoch_loss = 0
    
    while not done:
        input, traj, seq_length, done = Data.next()

        input = input.unsqueeze(0)
        
        output = model(input, seq_length)

        mask = torch.arange(Data.max_seq_length).unsqueeze(0) < seq_length
        mask = mask.unsqueeze(-1).expand_as(output)
        
        loss = criterion(output, traj.unsqueeze(0))
        masked_loss = (loss * mask.float()).sum() / mask.float().sum()
        
        optimizer.zero_grad()
        masked_loss.backward()
        optimizer.step()
        
        epoch_loss += masked_loss.item()
        batch += 1
    
    scheduler.step()
    avg_epoch_loss = epoch_loss / batch
    running_loss.append(avg_epoch_loss)
    print(f" Average Loss: {avg_epoch_loss}")

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': running_loss,
}, 'models/model.pth')

plt.figure(figsize=(10, 5))
plt.plot(running_loss)
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.show()