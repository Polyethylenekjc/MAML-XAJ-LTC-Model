import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src.factory.model_factory import ModelFactory
from src.model.FullFlowGrku import PEestimator

@ModelFactory.register("ct_full_flow_model")
class CTFullFlow(nn.Module):
    """
    CNN + Transformer + P-E 修正的完整径流预测模型
    """
    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        forecast_steps: int = 1,
        pe_hidden_dim: int = 64,
        pe_layers: int = 2,
    ):
        super(CTFullFlow, self).__init__()

        self.input_dim = input_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.forecast_steps = forecast_steps

        # -----------------------
        # CNN 提取局部时序特征
        # -----------------------
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(seq_length)
        self.conv_dropout = nn.Dropout(dropout)

        # -----------------------
        # Transformer 编码
        # -----------------------
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # -----------------------
        # 输出层
        # -----------------------
        self.fc_out = nn.Linear(hidden_dim, forecast_steps)

        # -----------------------
        # P-E 估计器
        # -----------------------
        self.pe_estimator = PEestimator(
            input_dim=3,  # 假设最后三个特征是 precipitation, evaporation, runoff
            hidden_dim=pe_hidden_dim,
            output_dim=1,
            num_layers=pe_layers
        )

    def forward(self, x):
        batch_size = x.size(0)

        # -----------------------
        # CNN 特征提取
        # -----------------------
        x_cnn = x.permute(0, 2, 1)  # [B, input_dim, seq_len]
        x_cnn = self.relu(self.conv1(x_cnn))
        x_cnn = self.conv_dropout(x_cnn)
        x_cnn = self.relu(self.conv2(x_cnn))
        x_cnn = self.pool(x_cnn)
        x_cnn = x_cnn.permute(0, 2, 1)  # [B, seq_len, hidden_dim]

        # -----------------------
        # Transformer 编码
        # -----------------------
        x_trans = self.input_projection(x_cnn)
        x_trans = self.transformer_encoder(x_trans)
        x_trans = x_trans.mean(dim=1)  # [B, hidden_dim]

        # -----------------------
        # 输出预测
        # -----------------------
        out = self.fc_out(x_trans)  # [B, forecast_steps]

        # -----------------------
        # P-E 修正
        # -----------------------
        if x.size(2) >= 3:
            pe_input = x[:, :, -3:]  # [B, seq_len, 3]
            pe_estimate = self.pe_estimator(pe_input)  # [B, seq_len, 1]
            pe_value = pe_estimate[:, -1, :]  # 使用最后时间步
            pe_value = pe_value.unsqueeze(1).expand(-1, self.forecast_steps, -1)  # [B, forecast_steps, 1]

            out = out.unsqueeze(-1)  # [B, forecast_steps, 1]
            out = out * pe_value  # 应用 P-E 修正
            out = out.view(batch_size, -1)  # [B, forecast_steps]

        return out

    @classmethod
    def from_config(cls, config):
        return cls(
            input_dim=config.get('input_dim', 10),
            seq_length=config.get('seq_length', 15),
            hidden_dim=config.get('hidden_dim', 64),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1),
            forecast_steps=config.get('forecast_steps', 1),
            pe_hidden_dim=config.get('pe_hidden_dim', 64),
            pe_layers=config.get('pe_layers', 2),
        )
