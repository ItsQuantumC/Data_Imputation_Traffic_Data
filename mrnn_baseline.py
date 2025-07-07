# mrnn_baseline.py

import pytorch_lightning as pl
import torch
import torch.nn as nn

from model_utils import BiGRU, initial_point_interpolation


class MRNN(pl.LightningModule):
    def __init__(self, dim, h_dim, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.dim = dim
        self.h_dim = h_dim
        self.learning_rate = learning_rate
        self.rnn_cells = nn.ModuleList([BiGRU(3, h_dim, 1) for _ in range(dim)])
        self.fc = nn.Sequential(
            nn.Linear(dim, dim), nn.Sigmoid(), nn.Linear(dim, dim), nn.Sigmoid()
        )

    def forward(self, x, m, t):
        rnn_imputed_x = self.rnn_predict(x, m, t)
        fc_imputed_x = self.fc(rnn_imputed_x)
        return fc_imputed_x
    
    def _calculate_loss(self, batch):
        x, m, t, _ = batch
        loss = 0
        for f in range(self.dim):
            rnn = self.rnn_cells[f]
            temp_input = torch.cat((x[:, :, f:f+1], m[:, :, f:f+1], t[:, :, f:f+1]), dim=2)
            forward_input = torch.zeros_like(temp_input)
            forward_input[:, 1:, :] = temp_input[:, :-1, :]
            backward_input = torch.flip(temp_input, [1])
            imputation = rnn(forward_input, backward_input)
            loss += torch.sqrt(torch.mean(torch.square(
                m[:, :, f:f+1] * imputation - m[:, :, f:f+1] * x[:, :, f:f+1]
            )))
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
    def rnn_predict(self, x, m, t):
        imputed_x_list = []
        for f in range(self.dim):
            rnn = self.rnn_cells[f]
            temp_input = torch.cat((x[:, :, f:f+1], m[:, :, f:f+1], t[:, :, f:f+1]), dim=2)
            forward_input = torch.zeros_like(temp_input)
            forward_input[:, 1:, :] = temp_input[:, :-1, :]
            backward_input = torch.flip(temp_input, [1])
            imputation = rnn(forward_input, backward_input)
            imputed_f = (1 - m[:, :, f:f+1]) * imputation + m[:, :, f:f+1] * x[:, :, f:f+1]
            imputed_x_list.append(imputed_f)
        imputed_x = torch.cat(imputed_x_list, dim=2)
        imputed_x = initial_point_interpolation(
            x.cpu().numpy(), m.cpu().numpy(), t.cpu().numpy(), imputed_x.cpu().numpy()
        )
        return torch.from_numpy(imputed_x).to(self.device)

    @torch.no_grad()
    def impute(self, x, m, t):
        self.eval()
        x_t, m_t, t_t = map(lambda a: torch.from_numpy(a).float().to(self.device), [x, m, t])
        rnn_imputed_x = self.rnn_predict(x_t, m_t, t_t)
        fc_imputed_x = self.fc(rnn_imputed_x)
        final_imputation = fc_imputed_x * (1 - m_t) + x_t * m_t
        final_imputation_np = final_imputation.cpu().numpy()
        final_imputation_np = initial_point_interpolation(
            x, m, t, final_imputation_np
        )
        return final_imputation_np
