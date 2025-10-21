import torch
import torch.nn as nn
from src.factory.model_factory import ModelFactory
from src.model.ltc_lnn import GRU_LTC


class PEestimator(nn.Module):
    """MLP P-E estimator: input [B, T, 3] -> output [B, T, 1]"""
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1, num_layers=2):
        super(PEestimator, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        out = self.mlp(x_flat)
        return out.view(B, T, -1)


@ModelFactory.register("full_flow_ltc_time_series_model")
class FullFlowGRULTC(nn.Module):
    """
    GRU 主干，最后输出阶段融合 LTC 的非线性单元。
    Forward returns [B, forecast_steps * output_dim]
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=1,
        num_layers=1,
        seq_length=None,
        forecast_steps=1,
        window_size=None,
        rtol=1e-6,
        atol=1e-6,
    ):
        super().__init__()
        self.window_size = window_size if window_size is not None else seq_length
        if self.window_size is None:
            raise ValueError("window_size or seq_length must be specified")

        self.forecast_steps = forecast_steps
        self.output_dim = output_dim

        # GRU 作为主干
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        # 全连接输出层
        self.fc_out = nn.Linear(hidden_dim, forecast_steps * output_dim)

        # LTC 作为非线性修正模块（只在最后融合用）
        self.ltc = GRU_LTC(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=1,
            rtol=rtol,
            atol=atol,
        )

        # P-E estimator
        self.pe_estimator = PEestimator(input_dim=3, hidden_dim=64, output_dim=1, num_layers=2)

    def forward(self, x):
        B = x.size(0)

        # 1. GRU 提取时序特征
        gru_out, h_n = self.gru(x)   # [B, T, H]
        h_last = h_n[-1]             # [B, H]

        # 2. 先做基本预测
        base_out = self.fc_out(h_last)   # [B, forecast_steps * output_dim]
        base_out = base_out.view(B, self.forecast_steps, self.output_dim)

        # 3. LTC 对 GRU 的输出做非线性修正
        #    将 h_last 看作 “产流过程的状态”
        ltc_out, _ = self.ltc(gru_out)   # [B, T, H]
        ltc_last = ltc_out[:, -1, :]     # [B, H]
        ltc_factor = torch.tanh(ltc_last)  # [-1,1] 作为非线性修正因子

        out = base_out + 0.1 * ltc_factor[:, None, :self.output_dim]

        # 4. 融合 P-E 估计
        if x.size(2) >= 3:
            pe_input = x[:, :, -3:]               # [B, T, 3]
            pe_est = self.pe_estimator(pe_input)  # [B, T, 1]
            pe_last = pe_est[:, -1, :]            # [B, 1]
            pe_vec = pe_last.unsqueeze(1).expand(-1, self.forecast_steps, -1)
            out = out * pe_vec

        return out.view(B, -1)

