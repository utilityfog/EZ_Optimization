import numpy as np

class EZSingleAssetEnv:
    """
    Single risky asset environment with Epstein Zin style consumption.

    We ignore portfolio weights, turnover and transaction costs.
    All post consumption wealth is placed in the risky asset.
    """

    def __init__(
        self,
        returns,          # array of gross returns, shape [T]
        features,         # array of features, shape [T, d]
        beta: float,
        psi: float,
        start_wealth: float = 1.0,
    ):
        assert len(returns) == len(features)
        self.returns = returns.astype(np.float32)
        self.features = features.astype(np.float32)
        self.T = len(returns)
        self.beta = beta
        self.psi = psi
        self.start_wealth = float(start_wealth)

        self.reset()

    def reset(self):
        self.t = 0
        self.W = self.start_wealth
        self.M = self.W
        return self._state()

    def _state(self):
        """
        State is [normalized wealth, features_t]
        """
        W_tilde = self.W / self.M if self.M > 0 else 1.0
        x_t = self.features[self.t]
        return np.concatenate([[W_tilde], x_t]).astype(np.float32)

    def step(self, c_t: float):
        """
        c_t is consumption rate in (0,1).
        Wealth update:
          C_t = c_t * W_t
          W_{t+1} = (W_t - C_t) * R_t
        External reward:
          r_ext_t = (1 - beta) * C_t^{1 - 1/psi}
        """
        c_t = float(np.clip(c_t, 1e-6, 1.0 - 1e-6))

        C_t = c_t * self.W
        R_t = float(self.returns[self.t])

        W_next = (self.W - C_t) * R_t
        W_next = max(W_next, 1e-8)

        self.W = W_next
        self.M = max(self.M, self.W)

        # EZ shaped external reward
        power = 1.0 - 1.0 / self.psi
        C_safe = max(C_t, 1e-8)
        r_ext = (1.0 - self.beta) * (C_safe ** power)

        self.t += 1
        done = self.t >= self.T - 1

        if not done:
            next_state = self._state()
        else:
            next_state = None

        info = {"W": self.W, "C": C_t}
        return next_state, r_ext, done, info