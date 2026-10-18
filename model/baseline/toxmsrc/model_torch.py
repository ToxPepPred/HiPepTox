from __future__ import annotations

import torch
from torch import nn


class ToxMSRCPyTorch(nn.Module):
    """PyTorch reimplementation of the original ToxMSRC architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.branch3 = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
        )

        self.pool = nn.MaxPool1d(kernel_size=3, stride=1)
        self.dropout_mid = nn.Dropout(0.4)
        self.conv_post = nn.Conv1d(128, 256, kernel_size=3, padding=0)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(45 * 256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout_fc = nn.Dropout(0.6)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, 49, 96]
        x = x.transpose(1, 2)  # [batch, 96, 49]

        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)
        x = x3 + x5 + x7

        x = self.pool(x)  # [batch, 128, 47]
        x = self.dropout_mid(x)
        x = self.conv_post(x)  # [batch, 256, 45]

        x_seq = x.transpose(1, 2)  # [batch, 45, 256]
        x_lstm, _ = self.lstm(x_seq)
        x = x_lstm + x_seq

        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = torch.relu(self.fc3(x))
        return self.output(x)
