import torch
import torch.nn as nn
import matplotlib.pyplot
import csv
import pandas as pd
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		self.hidden_size =  hidden_size
		self.i2h = nn.Linear(input_size + hidden_size, hidden_size) #input to hidden
		self.i2o = nn.Linear(input_size + hidden_size, output_size)
	def forward(self, input_tensor, hidden_tensor):
		combined = torch.cat((input_tensor, hidden_tensor),1)
		hidden = self.i2h(combined)
		output = self.i2o(combined)
		return output, hidden
	def init_hidden(self, batch_size=1):
		return torch.zeros(self.hidden_size,1 )
hyper_params = {
    'train_ratio': 0.8,
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 0.001,
	'hidden_size': 256,
	'input_size': 4,
	'output_size': 3
}
##load in the data
file_path = '/home/mainubuntu/Desktop/RBE577-MachineLearningforRobotics/homework3/data_array_smaller.csv'
df = pd.read_csv(file_path)
df.iloc[:, 1] += 1000  #fix the data because I messed it up

##Split the data into inputs and outputs and convert to tensors and segment the data to preserve the paths 
split_indices = df[(df['X'] == 0) & (df['Y'] == 0)].index.tolist()
split_indices.append(len(df))
segments = []
start_idx = 0
data_tensors =[]
    
train_ratio = hyper_params['train_ratio']  # 80% for training, 20% for validation
num_segments = len(split_indices)
print(num_segments)
num_train = int(train_ratio * num_segments)
num_val = num_segments-num_train
for end_idx in split_indices:
	segment = df.iloc[start_idx:end_idx]
	if not segment.empty:  # Check if the segment is not empty
		segments.append(segment)
	start_idx = end_idx+1  # Move to the next segment

segment = segments[1]
output_tensor = torch.tensor(segment[['X', 'Y', 'Z']].values, dtype=torch.float32, requires_grad=True)
input_tensor = torch.tensor(segment[['XDelta', 'YDelta', 'HeadingDelta', 'GammaDelta']].values, dtype=torch.float32)
# print(output_tensor)
#intiialize the loss function 
mse_loss = nn.MSELoss()
##initialize the RNN
rnn = RNN(hyper_params['input_size'], hyper_params['hidden_size'], hyper_params['output_size'])
hidden_tensor = rnn.init_hidden()
input_tensor = input_tensor.unsqueeze(1)  # Add a batch dimension
num_pads = hidden_tensor.shape[0]-input_tensor.shape[0]

print(hidden_tensor.shape)
print(input_tensor.shape)
optimizer = optim.Adam(rnn.parameters(), lr=0.001)
try:
	for epoch in range(hyper_params['num_epochs']):
		rnn.train()
		output, next_hidden = rnn(input_tensor, hidden_tensor)
		loss = mse_loss(output, output_tensor)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
except KeyboardInterrupt:
    print('\nTraining Interrupted')
# input_tensor = 

