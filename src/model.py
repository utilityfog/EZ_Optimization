import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCriticEZ(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_c = nn.Linear(hidden_dim, 1)
        self.log_std_c = nn.Parameter(torch.zeros(1))

        self.h_z = nn.Linear(hidden_dim, 1)
        self.h_y = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        h = self.body(s)
        mu = self.mu_c(h).squeeze(-1)
        std = F.softplus(self.log_std_c) + 1e-5
        z_hat = self.h_z(h).squeeze(-1)
        y_hat = self.h_y(h).squeeze(-1)
        return mu, std, z_hat, y_hat

    def act(self, s):
        mu, std, z_hat, y_hat = self.forward(s)
        eps = torch.randn_like(mu)
        y = mu + eps * std
        c = torch.sigmoid(y)

        log_norm = -0.5*((y-mu)/std)**2 - torch.log(std) - 0.5*np.log(2*np.pi)
        log_det = torch.log(c) + torch.log(1-c)
        logp = log_norm - log_det

        return c, logp, z_hat, y_hat