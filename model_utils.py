# model_utils.py

import numpy as np
import torch
import torch.nn as nn


def initial_point_interpolation(x, m, t, imputed_x):
    no, seq_len, dim = x.shape
    for i in range(no):
        for k in range(dim):
            for j in range(seq_len):
                if t[i, j, k] > j:
                    idx = np.where(m[i, :, k] == 1)[0]
                    if len(idx) > 0:
                        imputed_x[i, j, k] = x[i, np.min(idx), k]
    return imputed_x


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_layer_size, target_size):
        super().__init__()
        self.gru_f = nn.GRU(input_size, hidden_layer_size, batch_first=True)
        self.gru_b = nn.GRU(input_size, hidden_layer_size, batch_first=True)
        self.out = nn.Linear(hidden_layer_size * 2, target_size)
    def forward(self, f_input, b_input):
        f_output, _ = self.gru_f(f_input)
        b_output, _ = self.gru_b(b_input)
        concat_hidden = torch.cat((f_output, torch.flip(b_output, [1])), 2)
        return self.out(concat_hidden)
