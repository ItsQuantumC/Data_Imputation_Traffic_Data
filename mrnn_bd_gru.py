# mrnn_bd_gru.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class MRNN(pl.LightningModule):
    def __init__(self, dim, h_dim, learning_rate, n_layers=2):
        super().__init__()
        self.save_hyperparameters()
        self.dim = dim
        self.h_dim = h_dim
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.input_proj = nn.Linear(dim * 3, h_dim)
        self.rnn = nn.GRU(
            input_size=h_dim,
            hidden_size=h_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if n_layers > 1 else 0
        )
        self.output_proj = nn.Linear(h_dim * 2, dim)

    def forward(self, x, m, t):
        inp = torch.cat([x, m, t], dim=2)
        inp_proj = F.relu(self.input_proj(inp))
        rnn_out, _ = self.rnn(inp_proj)
        imputed_x = self.output_proj(rnn_out)
        return imputed_x

    def _calculate_loss(self, batch):
        x, m, t, _ = batch
        imputed_x = self.forward(x, m, t)
        loss = torch.sqrt(F.mse_loss(imputed_x[m > 0], x[m > 0]))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @torch.no_grad()
    def impute(self, x, m, t):
        self.eval()
        x_t = torch.from_numpy(x).float().to(self.device)
        m_t = torch.from_numpy(m).float().to(self.device)
        t_t = torch.from_numpy(t).float().to(self.device)
        imputed_x = self.forward(x_t, m_t, t_t)
        final_imputation = imputed_x * (1 - m_t) + x_t * m_t
        return final_imputation.cpu().numpy()
