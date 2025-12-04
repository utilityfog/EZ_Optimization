import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fracdiff import LearnableFracDiff1D


class ActorCriticEZ(nn.Module):
    """
    Shared trunk + three heads + learnable fractional differencing
    over a trailing window of simple returns.

    State layout assumed:

        [ W_norm,
          features_t (feature_dim),
          return_window (window_len) ]
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        use_fracdiff: bool = False,
        fracdiff_max_lag: int = 12,
        fracdiff_init_d: float = 0.4,
        window_len: int | None = None,
    ):
        super().__init__()

        self.use_fracdiff = use_fracdiff
        self.window_len = int(window_len) if window_len is not None else 0

        if self.use_fracdiff and self.window_len <= 0:
            raise ValueError("window_len must be > 0 when use_fracdiff=True")

        # feature_dim = total_dim - 1 (wealth) - window_len
        if self.window_len > 0:
            self.feature_dim = state_dim - 1 - self.window_len
            if self.feature_dim < 0:
                raise ValueError(
                    f"state_dim={state_dim} too small for window_len={self.window_len}"
                )
        else:
            self.feature_dim = state_dim - 1

        if use_fracdiff:
            self.fracdiff = LearnableFracDiff1D(
                max_lag=fracdiff_max_lag,
                init_d=fracdiff_init_d,
            )

        # shared trunk
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # policy head
        self.mu_c = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.mu_c.bias, -2.0)
        self.log_std_c = nn.Parameter(torch.zeros(1))

        # critics
        self.z_head = nn.Linear(hidden_dim, 1)
        self.y_head = nn.Linear(hidden_dim, 1)

    def _apply_fracdiff(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable FD to the trailing return window portion of the state.
        Input:  state [B, state_dim]
        Output: same shape, with window part transformed.
        """
        if not self.use_fracdiff or self.window_len <= 0:
            return state

        # split state
        W = state[:, :1] # [B, 1]
        feats = state[:, 1 : 1 + self.feature_dim] # [B, feature_dim]
        window = state[:, 1 + self.feature_dim :] # [B, window_len]

        # FD over window, same shape [B, window_len]
        fd_window = self.fracdiff(window)
        
        # Replace any nans in fd_window (Is this the right approach?)
        fd_window = torch.nan_to_num(fd_window, nan=0.0, posinf=1.0, neginf=-1.0)

        # reassemble
        return torch.cat([W, feats, fd_window], dim=1)

    def forward(self, state):
        """
        state: [B, state_dim]
        returns:
          mu:     [B]
          std:    [B]
          z_hat:  [B]
          y_hat:  [B]
        """
        fd_state = self._apply_fracdiff(state)
        fd_state = fd_state - fd_state.mean(dim=1, keepdim=True)
        fd_state = fd_state / (fd_state.std(dim=1, keepdim=True) + 1e-3)
        fd_state = torch.nan_to_num(fd_state, nan=0.0, posinf=1.0, neginf=-1.0)

        h = self.body(fd_state)
        
        # policy (μ & σ)
        mu = self.mu_c(h).squeeze(-1) # [B]
        std = F.softplus(self.log_std_c).expand_as(mu) + 1e-5 # [B]

        # critics
        z_hat = self.z_head(h).squeeze(-1) # [B]
        y_raw = self.y_head(h).squeeze(-1) # [B]
        y_hat = torch.clamp(F.softplus(y_raw), 1e-6, 1e2) # keep >0

        return mu, std, z_hat, y_hat

    def act(self, state):
        """
        Sample an action and compute its log probability.

        state: [state_dim] or [B, state_dim]

        Returns:
          c:     [B], sampled consumption rate in (0,1)
          logp:  [B], log prob of c
          z_hat: [B]
          y_hat: [B]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        mu, std, z_hat, y_hat = self.forward(state)
        
        # print("INITIAL MU:", mu.detach().cpu().numpy()[0])

        μ = torch.clamp(mu, -20.0, 20.0)
        σ = torch.clamp(std, 1e-3, 5.0)

        eps = torch.randn_like(μ)
        y   = μ + eps * σ
        c   = torch.sigmoid(y)
        c_safe = torch.clamp(c, 1e-6, 1.0 - 1e-6)

        log_norm = (
            -0.5 * ((y - μ) / σ) ** 2
            - torch.log(σ)
            - 0.5 * math.log(2.0 * math.pi)
        )

        log_det = torch.log(c_safe) + torch.log(1.0 - c_safe)
        logp = (log_norm - log_det).squeeze(-1)

        return c.squeeze(-1), logp, z_hat.squeeze(-1), y_hat.squeeze(-1)