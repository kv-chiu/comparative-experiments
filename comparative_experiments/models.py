import torch
import torch.nn as nn


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class MyCNN(nn.Module):
    """
    A simple CNN model for classification.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_size : int
        Number of classes to predict.

    Methods
    ----------
    forward(x)
        Defines the forward pass of the model.

    Input
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, input_channels, sequence_length).

    Output
    ----------
    x : torch.Tensor
        Output tensor of shape (batch_size, output_size).
    """

    def __init__(self, input_channels, output_size):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
