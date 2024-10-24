import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn

# data = pd.read_csv('data/dataArray_table.csv', nrows=1000)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.rnn(x)  # RNN outputs and hidden state
        out = self.fc(out[:, -1, :])  # Use the output of the last time step
        return out

def plotTrajTensor(data):
    data_np = data.detach().numpy()
    plt.figure()
    plt.plot(data_np[0,:,1], data_np[0,:,2], '.')
    plt.title('X vs Y')
    plt.show()

class DataClass:
    def __init__(self, data):
        data.iloc[:,1] += 1000
        data['isNew'] = data['X'] == 0
        data['group_id'] = data['isNew'].cumsum() - 1
        self.data = data
        self.current_idx = 0
        self.n_runs = max(data['group_id'])

    def __len__(self):
        return len(self.data)
    
    def next(self):
        run = self.get_run_torch(self.current_idx)
        self.current_idx += 1
        if self.current_idx >= self.n_runs:
            done = True
        else:
            done = False
        return run, done

    def get_run_torch(self, idx):
        return torch.tensor(self.data.loc[self.data['group_id'] == idx, ['X', 'Y', 'Z', 'XDelta', 'YDelta', 'HeadingDelta', 'GammaDelta']].values, dtype=torch.float)

    def reset(self):
        self.current_idx = 0

# Data = DataClass(data)

# done = False
# while not done:
#     run, done = Data.next()
#     print(f"Last Idx: {Data.current_idx}, shape: {run.shape}")