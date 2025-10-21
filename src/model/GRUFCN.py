import torch
import torch.nn as nn
from ..factory.model_factory import ModelFactory

@ModelFactory.register("gru_fcn")
class GRUFCNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        forecast_steps: int = 1
    ):
        super(GRUFCNModel, self).__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.forecast_steps = forecast_steps

        # Step 1: GRU branch for sequential pattern learning
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Step 2: FCN branch for feature extraction using dilated convolutions
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=8, padding='same')
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same')
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same')
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.relu = nn.ReLU()
        self.conv_dropout = nn.Dropout(dropout)

        # Step 3: Global max pooling for the convolutional branch
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Step 4: Final fully connected layer for prediction
        # Combining both branches: GRU last hidden state + FCN features
        self.fc_out = nn.Linear(hidden_dim + 128, forecast_steps)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # GRU branch
        gru_out, h_n = self.gru(x)  # gru_out: (batch, seq_len, hidden_dim)
        gru_features = h_n[-1]  # Last layer's final hidden state: (batch, hidden_dim)

        # FCN branch
        x_permuted = x.permute(0, 2, 1)  # -> (batch, input_dim, seq_length)
        
        # Dilated convolution layers
        conv_out = self.conv_dropout(self.relu(self.bn1(self.conv1(x_permuted))))
        conv_out = self.conv_dropout(self.relu(self.bn2(self.conv2(conv_out))))
        conv_out = self.conv_dropout(self.relu(self.bn3(self.conv3(conv_out))))
        
        # Global max pooling
        fcn_features = self.global_max_pool(conv_out).squeeze(-1)  # (batch, 128)

        # Concatenate features from both branches
        combined_features = torch.cat([gru_features, fcn_features], dim=1)  # (batch, hidden_dim + 128)

        # Final prediction
        output = self.fc_out(combined_features)  # (batch, forecast_steps)
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