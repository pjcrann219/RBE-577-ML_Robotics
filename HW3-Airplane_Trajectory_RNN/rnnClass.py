import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_seq_length,num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        self.input_fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x, seq_length):
        batch_size = x.size(0)
        
        x = self.input_fc(x)
        
        x = x.unsqueeze(1).repeat(1, self.max_seq_length, 1)
        
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(x.device) * 0.1
        
        out, hidden = self.rnn(x, h0)
        
        out = self.fc(out)
        
        return out
