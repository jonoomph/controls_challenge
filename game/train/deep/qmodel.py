import torch
import torch.nn as nn


class QSteeringNet(nn.Module):
    def __init__(self, input_size=15, hidden_size=80, num_layers=2, window_size=7, action_space=257):
        super(QSteeringNet, self).__init__()
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.action_space = action_space
        self.num_layers = num_layers

        # LSTM to process the time-series input
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Final layer outputs Q-values for each discrete action
        self.fc3 = nn.Linear(hidden_size, action_space)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0)

    def forward(self, x):
        # x: (batch, window_size, input_size)
        lstm_out, _ = self.lstm(x)
        # Use the output from the last time step
        x = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
