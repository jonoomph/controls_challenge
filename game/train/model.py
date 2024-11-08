import torch
import torch.nn as nn


class PIDControllerNet(nn.Module):
    def __init__(self, hidden_size=85, num_layers=2, window_size=7, input_size=25):
        super(PIDControllerNet, self).__init__()
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Define LSTM and fully connected layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Assuming x has shape (batch_size, window_size, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out has shape (batch_size, window_size, hidden_size)
        x = lstm_out[:, -1, :]  # Take the output from the last time step

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
