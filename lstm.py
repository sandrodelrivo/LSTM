import torch
import torch.nn as nn


class LSTMXOR(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMXOR, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x, lengths=True):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out_lstm, _ = self.lstm(x, (h0, c0))
        out = self.fc(out_lstm)

        predictions = self.activation(out)
        return predictions
