import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from DataClass import DataClass
from rnnClass import RNN
from sklearn.model_selection import train_test_split
import datetime
import os
from torch.utils.tensorboard import SummaryWriter
def create_tensorboard_writer():
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', current_time)
    return SummaryWriter(log_dir) 
input_size = 4
hidden_size = 32
output_size = 3
learning_rate = 0.0005
num_epochs = 1000
num_layers = 3
test_percent = 0.2
seed = 25 #random gen seed

##Load and Split the data
# Initialize TensorBoard writer
writer = create_tensorboard_writer()

# Log hyperparameters
writer.add_hparams(
    {
        'learning_rate': learning_rate,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'batch_size': 1  # Since we're processing one sequence at a time
    },
    {'dummy': 0}  # Required placeholder metric
)

data = pd.read_csv('data/dataArray_table.csv', nrows=10000)
data.iloc[:,1] += 1000
data['isNew'] = data['X'] == 0
data['group_id'] = data['isNew'].cumsum() - 1
groups = data['group_id'].unique()
groups = data['group_id'].unique()
train, test = train_test_split(groups, test_size = test_percent, random_state=seed)
train_data = data[data['group_id'].isin(train)].copy()
test_data = data[data['group_id'].isin(test)].copy()
train_data['group_id'] = pd.Categorical(train_data['group_id']).codes
test_data['group_id'] = pd.Categorical(test_data['group_id']).codes
train_loader = DataClass(train_data)
test_loader = DataClass(test_data)
train_losses = []
test_losses = []
Data = DataClass(data)




##initialize the Model
model = RNN(input_size, hidden_size, output_size, Data.max_seq_length, num_layers)
criterion = nn.MSELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

running_loss = []
sample_input = torch.randn(1, input_size)
writer.add_graph(model, (sample_input, train_loader.max_seq_length))
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}", end='')
    Data.reset()
    done = False
    batch = 0
    epoch_loss = 0
    model.train()
    while not done:
        input, traj, seq_length, done = train_loader.next()

        input = input.unsqueeze(0)
        
        output = model(input, seq_length)

        mask = torch.arange(Data.max_seq_length).unsqueeze(0) < seq_length
        mask = mask.unsqueeze(-1).expand_as(output)
        
        loss = criterion(output, traj.unsqueeze(0))
        masked_loss = (loss * mask.float()).sum() / mask.float().sum()
        
        optimizer.zero_grad()
        masked_loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'gradients/{name}', param.grad, epoch)
        optimizer.step()
        
        epoch_loss += masked_loss.item()
        batch += 1
    
    scheduler.step()
    avg_epoch_loss = epoch_loss / batch
    running_loss.append(avg_epoch_loss)

    ##test loop
    model.eval()
    total_loss_test = 0
    num_batches = 0
    with torch.no_grad():
        test_loader.reset()
        done = False
        while not done:
            input, traj, seq_length, done = test_loader.next()
            input = input.unsqueeze(0)
            output = model(input, seq_length)
            
            mask = torch.arange(test_loader.max_seq_length).unsqueeze(0) < seq_length
            mask = mask.unsqueeze(-1).expand_as(output)
            
            loss = criterion(output, traj.unsqueeze(0))
            masked_loss = (loss * mask.float()).sum() / mask.float().sum()
            
            total_loss_test += masked_loss.item()
            num_batches += 1


    avg_test_loss = total_loss_test/num_batches
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/test', avg_test_loss, epoch)
    writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)        
    for name, param in model.named_parameters():
        writer.add_histogram(f'parameters/{name}', param.data, epoch)
    print(f"Epoch: {epoch} Train Loss: {avg_epoch_loss:.6f} Test Loss: {avg_test_loss:.6f}")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': running_loss,
}, 'models/model.pth')
writer.close()
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()