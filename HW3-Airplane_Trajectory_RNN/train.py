import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from DataClass import DataClass
from rnnClass import RNN
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

name = datetime.now().strftime("%y%m%d%H%M%S")
print(f"Run {name}")
writer = SummaryWriter('runs/run_' + name)

data = pd.read_csv('data/dataArray_table.csv', nrows=250000)
Data = DataClass(data)

input_size = 4
hidden_size = 32
output_size = 3
learning_rate = 0.0005
num_epochs = 400

model = RNN(input_size, hidden_size, output_size, Data.max_seq_length)

# Used to continue training
# checkpoint = torch.load('models/model_cont_dev_1.pth')
# model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.MSELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

running_train_loss = []
running_test_loss = []
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}", end='')

    ## Training
    Data.reset()
    done = False
    train_batch_length = 0
    train_epoch_loss = 0
    model.train()
    while not done:
        if Data.current_idx not in Data.training_group_ids:
            # id not in training group, skip
            done = Data.iterate_idx()
        else:
            # id in training group, train
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
            
            train_epoch_loss += masked_loss.item()
            train_batch_length += 1

    ## Testing
    Data.reset()
    done = False
    test_batch_length = 0
    test_epoch_loss = 0
    model.eval()
    while not done:
        if Data.current_idx not in Data.testing_group_ids:
            # id not in testing group, skip
            done = Data.iterate_idx()
        else:
            # id in testing group, train
            with torch.no_grad():
                input, traj, seq_length, done = Data.next()

                input = input.unsqueeze(0)
                
                output = model(input, seq_length)

                mask = torch.arange(Data.max_seq_length).unsqueeze(0) < seq_length
                mask = mask.unsqueeze(-1).expand_as(output)
                
                loss = criterion(output, traj.unsqueeze(0))
                masked_loss = (loss * mask.float()).sum() / mask.float().sum()
                
                test_epoch_loss += masked_loss.item()
                test_batch_length += 1

    scheduler.step()
    avg_train_epoch_loss = train_epoch_loss / train_batch_length
    running_train_loss.append(avg_train_epoch_loss)

    avg_test_epoch_loss = test_epoch_loss / test_batch_length
    running_test_loss.append(avg_test_epoch_loss)
    print(f" Train Loss: {avg_train_epoch_loss}, Test Loss: {avg_test_epoch_loss}")

    # Log to tensorboard
    writer.add_scalar('Loss/train', avg_train_epoch_loss, epoch)
    writer.add_scalar('Loss/test', avg_test_epoch_loss, epoch)
    writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': running_train_loss,
}, 'models/model.pth')

plt.figure(figsize=(10, 5))
plt.plot(running_train_loss, 'b',label='Training Loss')
plt.plot(running_test_loss, 'r', label='Testing Loss')
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.show()
writer.close()