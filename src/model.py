import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ActorCriticEZ(nn.Module):
    """
    MLP with shared trunk and three heads:
      - policy head for c_t (consumption rate)
      - z head approximating z(V_t)
      - y head approximating y(V_t) for use at t+1
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # policy: Gaussian in pre-sigmoid space
        self.mu_c = nn.Linear(hidden_dim, 1)
        self.log_std_c = nn.Parameter(torch.zeros(1))

        # critics
        self.z_head = nn.Linear(hidden_dim, 1)
        self.y_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        h = self.body(state)
        mu = self.mu_c(h).squeeze(-1)

        std = torch.softplus(self.log_std_c) + 1e-5

        z_hat = self.z_head(h).squeeze(-1)
        y_hat = self.y_head(h).squeeze(-1)
        return mu, std, z_hat, y_hat

    def act(self, state):
        """
        Input: state tensor shape [batch] or [batch, dim]
        Output:
          c_t in (0,1)
          log_prob of action
          z_hat, y_hat
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        mu, std, z_hat, y_hat = self.forward(state)
        eps = torch.randn_like(mu)
        y = mu + eps * std
        c = torch.sigmoid(y)

        # log prob under Gaussian then change of variable through sigmoid
        log_norm = -0.5 * ((y - mu) / std) ** 2 - torch.log(std) - 0.5 * math.log(
            2.0 * math.pi
        )
        log_det = torch.log(c) + torch.log(1.0 - c)
        logp = (log_norm - log_det).squeeze(-1)

        return c.squeeze(-1), logp, z_hat.squeeze(-1), y_hat.squeeze(-1)