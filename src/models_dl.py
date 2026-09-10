import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1D-CNN over the time axis: Conv1d expects (batch, channels, time)."""

    def __init__(self, n_attrs: int, n_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_attrs, n_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_channels),
            nn.ReLU(),
            nn.Conv1d(n_channels, n_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, attrs) -> (batch, attrs, time)
        x = x.transpose(1, 2)
        x = self.net(x)
        return self.head(x).squeeze(-1)


class LSTMClassifier(nn.Module):
    """LSTM over the time axis, using the final hidden state for classification."""

    def __init__(self, n_attrs: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_attrs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, attrs)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.head(last_hidden).squeeze(-1)
