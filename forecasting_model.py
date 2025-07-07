# forecasting_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src):
        outputs, hidden = self.rnn(src)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear((hidden_dim * 2) + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.rnn = nn.GRU(
            (hidden_dim * 2) + output_dim,
            hidden_dim,
            n_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear((hidden_dim * 2) + hidden_dim + output_dim, output_dim)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((input, weighted), dim=2)
        gru_input_hidden = hidden.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)

        output, gru_output_hidden = self.rnn(rnn_input, gru_input_hidden)

        input = input.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        prediction = self.fc_out(torch.cat((output, weighted, input), dim=1))
        return prediction, gru_output_hidden[-1]


class Seq2Seq(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.attention = Attention(hidden_dim)
        self.encoder = Encoder(input_dim, hidden_dim, n_layers, dropout)
        self.decoder = Decoder(output_dim, hidden_dim, n_layers, dropout, self.attention)
        self.learning_rate = learning_rate

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len, _ = trg.shape
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[:, 0, :]
        for t in range(trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t, :] if teacher_force else output
        return outputs

    def training_step(self, batch, batch_idx):
        src, trg = batch
        output = self.forward(src, trg)
        loss = F.mse_loss(output, trg)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg = batch
        output = self.forward(src, trg, 0)
        loss = F.mse_loss(output, trg)
        self.log('val_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Generates predictions for a batch."""
        src, trg = batch
        output = self.forward(src, trg, 0)
        return {'predictions': output, 'ground_truth': trg}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
