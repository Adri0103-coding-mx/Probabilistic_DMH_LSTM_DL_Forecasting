import torch
import torch.nn as nn


class Attention_LSTM(nn.Module):
    """
    Minimal LSTM model for direct multi-horizon forecasting.

    The model maps an input sequence X of shape (B, L, F) to a direct
    multi-step forecast of shape (B, H, O), where:
        B = batch size
        L = lookback window length
        F = number of input features
        H = forecast horizon
        O = number of target variables

    This minimal version uses the last hidden state of the LSTM as the
    temporal representation and a single linear readout layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        horizon: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.horizon = horizon
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Single-head direct projection: (B, hidden_size) → (B, H * O)
        self.linear = nn.Linear(hidden_size, horizon * output_size)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, F).

        Returns
        -------
        y_hat : torch.Tensor
            Direct multi-horizon forecast of shape (B, H, O).
        """
        # LSTM encoding
        _, (h_n, _) = self.lstm(x)

        # Last layer hidden state: (B, hidden_size)
        h_last = h_n[-1]

        # Linear projection and reshape
        out = self.linear(h_last)
        y_hat = out.view(out.size(0), self.horizon, self.output_size)

        return y_hat
