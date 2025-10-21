import torch
import torch.nn as nn
import torch.nn.functional as F
from ..factory.model_factory import ModelFactory


class TemporalAttention(nn.Module):
    """时间注意力模块：捕捉输入序列中时间步之间的依赖"""
    def __init__(self, input_dim, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_weights = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        out = torch.bmm(attn_weights, V)
        return out, attn_weights


class PeriodicAttention(nn.Module):
    """周期注意力模块：基于周期索引增强特征（例如每7天、30天或12个月）"""
    def __init__(self, input_dim, hidden_dim, period=7):
        super(PeriodicAttention, self).__init__()
        self.period = period
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        batch_size, seq_length, input_dim = x.shape

        # 构建周期索引，例如 period=7 时，索引为 [0,1,2,3,4,5,6,0,1,...]
        periodic_index = torch.arange(seq_length, device=x.device) % self.period
        periodic_embedding = F.one_hot(periodic_index, num_classes=self.period).float()
        periodic_embedding = periodic_embedding.unsqueeze(0).repeat(batch_size, 1, 1)

        # 将周期embedding映射到与x相同维度后融合
        periodic_feature = torch.cat([x, periodic_embedding], dim=-1)
        Q = self.query(periodic_feature)
        K = self.key(periodic_feature)
        V = self.value(periodic_feature)

        attn_weights = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        out = torch.bmm(attn_weights, V)
        return out, attn_weights


@ModelFactory.register("tpt_model")
class TPTModel(nn.Module):
    """TPT模型：融合时间与周期注意力机制"""
    def __init__(self, input_dim, output_dim, seq_length, forecast_steps=1, hidden_dim=64, period=7):
        super(TPTModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.forecast_steps = forecast_steps
        self.hidden_dim = hidden_dim
        self.period = period

        # 编码层（初步映射）
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 时间注意力 & 周期注意力
        self.temporal_attn = TemporalAttention(hidden_dim, hidden_dim)
        self.periodic_attn = PeriodicAttention(hidden_dim + period, hidden_dim, period=period)

        # 融合层
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

        # 解码层（预测层）
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, forecast_steps * output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, input_dim)
        Returns:
            (batch_size, forecast_steps * output_dim)
        """
        x = self.input_proj(x)

        temporal_out, _ = self.temporal_attn(x)
        periodic_out, _ = self.periodic_attn(x)

        fused = torch.cat([temporal_out, periodic_out], dim=-1)
        fused = self.fusion(fused)

        pooled = torch.mean(fused, dim=1)
        output = self.fc_out(pooled)
        return output

    @classmethod
    def from_config(cls, config):
        params = config.get('params', {})
        return cls(
            input_dim=params.get('input_dim', 10),
            output_dim=params.get('output_dim', 1),
            seq_length=params.get('seq_length', 15),
            forecast_steps=params.get('forecast_steps', 1),
            hidden_dim=params.get('hidden_dim', 64),
            period=params.get('period', 7)
        )
