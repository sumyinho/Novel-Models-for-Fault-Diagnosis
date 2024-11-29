import torch
import torch.nn as nn
import torch.nn.functional as F


# Discriminative Sparse Autoencoder for Gearbox Fault Diagnosis Toward Complex Vibration Signals
# DOI: 10.1109/TIM.2022.3203440
class DSAE(nn.Module):

    def __init__(self, in_units, hidden_units, sparsity_constraint=.1, device='cuda:0'):
        super(DSAE, self).__init__()

        self.device = device
        self.sparsity_constraint = torch.tensor([sparsity_constraint]).to(self.device)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_units, hidden_units),
            nn.ReLU(),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_units, in_units),
            nn.Sigmoid(),
        )

    # def get_kl_loss(self, hidden):
    #     hidden = hidden.squeeze()
    #     print("hidden.shape:", hidden.shape)
    #     avg_activation = torch.mean(hidden, dim=0)
    #     print("avg_activation:", avg_activation.shape)
    #     return self.kl_loss(torch.log(self.sparsity_constraint),
    #                         torch.mean(avg_activation, dim=0))

    # def get_sparse_loss(self, hidden):
    #     rho_hat = torch.mean(hidden, dim=0)
    #     rho = torch.ones_like(rho_hat) * self.sparsity_constraint
    #     kl_div = self.kl_loss(torch.log(rho), rho_hat)
    #     # print("rho_hat:", rho_hat.shape)
    #     # print("rho:", rho.shape)
    #     return kl_div

    def get_sparse_loss(self, hidden):
        rho_hat = torch.mean(hidden, dim=0)
        rho = torch.ones_like(rho_hat) * self.sparsity_constraint
        kl_div = self.kl_loss(torch.log_softmax(rho_hat, dim=0), torch.softmax(rho, dim=0))
        return kl_div

    def forward(self, x):
        hidden = self.encoder(x)
        output = self.decoder(hidden)
        return output, hidden


class DSAE_Discriminator(nn.Module):

    def __init__(self, in_units, out_units, dropout=.1):
        super(DSAE_Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(in_units, out_units),
        )

    def forward(self, x):
        output = self.discriminator(x)
        return F.softmax(output, dim=-1)
