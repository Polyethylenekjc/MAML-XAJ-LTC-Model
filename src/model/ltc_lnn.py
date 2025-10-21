# src/model/gru_ltc.py
import torch
import torch.nn as nn

class LTCFunc(nn.Module):

    def __init__(self, input_dim, hidden_dim, tau_init=5.0):
        super(LTCFunc, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.tau = nn.Parameter(torch.ones(hidden_dim) * tau_init)
        self.W_in = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, h, x_t):
        Wx = self.W_in(x_t)
        Wh = self.W_rec(h)

        drive = torch.tanh(Wx + Wh)  
        dhdt = -(1.0 / self.tau) * h + drive
        return dhdt


class LTCStep(nn.Module):
    def __init__(self, input_dim, hidden_dim, dt=1.0, tau_init=5.0):
        super(LTCStep, self).__init__()
        self.func = LTCFunc(input_dim, hidden_dim, tau_init=tau_init)
        self.dt = dt

    def forward(self, h0, x_t):
        dhdt = self.func(h0, x_t)
        h1 = h0 + self.dt * dhdt 

        if torch.isnan(h1).any() or torch.isinf(h1).any():
            h1 = torch.zeros_like(h1)

        return h1


class GRU_LTC_Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 solver='dopri5', rtol=1e-3, atol=1e-3, eps=1e-2):
        super(GRU_LTC_Cell, self).__init__()
        self.hidden_dim = hidden_dim

        self.W_xz = nn.Linear(input_dim, hidden_dim)
        self.W_hz = nn.Linear(hidden_dim, hidden_dim)
        self.W_xr = nn.Linear(input_dim, hidden_dim)
        self.W_hr = nn.Linear(hidden_dim, hidden_dim)

        self.ltc_step = LTCStep(input_dim, hidden_dim)

    def forward(self, x_t, h_prev):
        z_t = torch.sigmoid(self.W_xz(x_t) + self.W_hz(h_prev))
        r_t = torch.sigmoid(self.W_xr(x_t) + self.W_hr(h_prev))

        h_reset = r_t * h_prev
        h_tilde = self.ltc_step(h_reset, x_t)

        h_new = (1 - z_t) * h_prev + z_t * h_tilde
        return h_new


class GRU_LTC(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1,
                 solver='dopri5', rtol=1e-3, atol=1e-3, eps=1e-2):
        super(GRU_LTC, self).__init__()
        self.layers = nn.ModuleList([
            GRU_LTC_Cell(input_dim if i == 0 else hidden_dim,
                         hidden_dim,
                         rtol=rtol, atol=atol, eps=eps)
            for i in range(num_layers)
        ])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, h0=None):
        B, T, _ = x.shape
        if h0 is None:
            h0 = x.new_zeros(self.num_layers, B, self.hidden_dim)

        hs = []
        out = x
        for layer_idx, cell in enumerate(self.layers):
            h = h0[layer_idx]
            outputs = []
            for t in range(T):
                h = cell(out[:, t, :], h)
                outputs.append(h.unsqueeze(1))
            out = torch.cat(outputs, dim=1)
            hs.append(h.unsqueeze(0))

        h_final = torch.cat(hs, dim=0)
        return out, h_final
