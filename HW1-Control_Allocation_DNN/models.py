import torch
import torch.nn as nn


class allocationNN(nn.Module):
    def __init__(self, dropout):
        super(allocationNN, self).__init__()
        self.name = 'allocationNN'
       
        self.encoder = nn.Sequential(
            nn.Linear(3, 24),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(12, 5)
        )

        self.decoder = nn.Sequential(
            nn.Linear(5, 24),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(12, 3)
        )

    def forward(self, input):
        u = self.encoder(input)
        tau = self.decoder(u)
        
        return u, tau

class allocationNN_LSTM(nn.Module):
    def __init__(self):
        super(allocationNN, self).__init__()
       
        self.encoder_lstm1 = nn.LSTM(3, 64, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.encoder_fc = nn.Linear(64, 5)

        self.decoder_lstm1 = nn.LSTM(5, 64, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(64, 64, batch_first=True)
        self.decoder_fc = nn.Linear(64, 3)

    def forward(self, input):
        x, (hidden, cell) = self.encoder_lstm1(input)
        x, (hidden, cell) = self.encoder_lstm2(x)
        u = self.encoder_fc(x[:, -1, :])

        u = u.unsqueeze(1)
        
        x, _ = self.decoder_lstm1(u)
        x, _ = self.decoder_lstm2(x)
        tau = self.decoder_fc(x[:, -1, :])
        
        return u, tau
        
    

# model = allocationNN()
# input_data = torch.randn(10, 5, 3)
# print(input_data)
# u, tau = model.forward(input_data)
# print(u)
# print(tau)