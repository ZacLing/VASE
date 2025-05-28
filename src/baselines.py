import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=256,
                 num_layers=3,
                 rnn='rnn'):
        super(BaseModel, self).__init__()

        self.in_proj = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.out_proj = nn.Linear(in_features=hidden_dim, out_features=input_dim)
        if rnn == 'rnn':
            self.rnn = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers)
        elif rnn == 'lstm':
            self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers)
        elif rnn == 'mlp':
            self.rnn = nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
            )

    def forward(self, x):
        out = self.out_proj(self.rnn(self.in_proj(x))[0])
        return out


class Baseline(nn.Module):
    def __init__(self,
                 base_models):
        super(Baseline, self).__init__()
        self.models = nn.ModuleDict(base_models)

    def forward(self, x_dict):
        out_dict = {}
        for key, x in x_dict.items():
            out_dict[key] = self.models[key](x["data"])
        return out_dict, None, None