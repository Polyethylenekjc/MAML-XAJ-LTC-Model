import torch
import torch.nn as nn
import torch.nn.functional as F
from ..factory.model_factory import ModelFactory

@ModelFactory.register("cnn_gru_attention")
class CNNGRUAttentionModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        forecast_steps: int = 1
    ):
        super(CNNGRUAttentionModel, self).__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.forecast_steps = forecast_steps

        # Step 1: Use 1D-CNN to extract local temporal features
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(output_size=seq_length)  # Maintain sequence length
        self.conv_dropout = nn.Dropout(dropout)

        # Step 2: Project to GRU dimension (if needed)
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)

        # Step 3: Replace Transformer with GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True  # Input/output shape: (batch, seq, feature)
        )

        # Step 4: Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Step 5: Output layer: map to multi-step prediction
        self.fc_out = nn.Linear(hidden_dim, forecast_steps)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x = x.permute(0, 2, 1)  # -> (batch, input_dim, seq_length)

        # CNN feature extraction
        x = self.relu(self.conv1(x))  # (batch, hidden_dim, seq_length)
        x = self.conv_dropout(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # Optional: stabilize length
        x = x.permute(0, 2, 1)  # -> (batch, seq_length, hidden_dim)

        # Projection + GRU
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)
        x, h_n = self.gru(x)  # x: (batch, seq_len, hidden_dim), h_n: (num_layers, batch, hidden_dim)

        # Apply attention
        attn_output, _ = self.attention(x, x, x)  # (batch, seq_len, hidden_dim)

        # Global average pooling to utilize information from the entire sequence
        x = attn_output.mean(dim=1)  # (batch, hidden_dim)

        # Output prediction
        output = self.fc_out(x)  # (batch, forecast_steps)
        return output

    @classmethod
    def from_config(cls, config):
        input_dim = config.get('input_dim', 10)
        seq_length = config.get('seq_length', 15)
        hidden_dim = config.get('hidden_dim', 64)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.1)
        forecast_steps = config.get('forecast_steps', 1)

        return cls(
            input_dim=input_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            forecast_steps=forecast_steps
        )