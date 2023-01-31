import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=50, output_size=3, num_layers=3):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size,
                            batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output


class GRU(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=50, output_size=3, num_layers=3):
        super(GRU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_layer_size,
                          batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[:, -1, :])
        return output
