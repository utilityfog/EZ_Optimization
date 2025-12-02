import numpy as np

class EZSingleAssetEnv:
    """
    Single risky asset environment for Epstein Zin consumption timing.

    Action: c_t, the fraction of wealth consumed in (0,1).
    Wealth dynamics:
        C_t = c_t * W_t
        W_{t+1} = (W_t - C_t) * R_t
    External reward:
        r_ext_t = (1 - beta) * C_t^{1 - 1/psi}

    All remaining wealth is automatically invested in the risky asset.
    There are no portfolio weights, turnover costs, or transaction costs.
    """

    def __init__(self, returns, features, beta, psi, start_wealth=1.0):
        """
        returns: array [T] of gross returns R_t
        features: array [T, d] of feature vectors
        """
        assert len(returns) == len(features)

        self.returns = returns.astype(np.float32)
        self.features = features.astype(np.float32)
        self.T = len(returns)

        self.beta = float(beta)
        self.psi = float(psi)
        self.start_wealth = float(start_wealth)

        self.reset()

    def reset(self):
        self.t = 0
        self.W = self.start_wealth
        self.M = self.W     # running max for normalized wealth
        return self._state()

    def _state(self):
        """
        State = [normalized wealth, feature_t]
        where normalized wealth = W_t / max past wealth.
        """
        W_norm = self.W / self.M if self.M > 0 else 1.0
        x_t = self.features[self.t]
        return np.concatenate([[W_norm], x_t]).astype(np.float32)

    def step(self, c_t):
        """
        c_t in (0,1). Clip to avoid undefined logit values.
        Computes C_t, W_{t+1}, and EZ external reward.
        """

        c_t = float(np.clip(c_t, 1e-6, 1.0 - 1e-6))

        # consumption
        C_t = c_t * self.W

        # return for this period
        R_t = float(self.returns[self.t])

        # wealth update
        W_next = (self.W - C_t) * R_t
        W_next = max(W_next, 1e-8)  # avoid numerical collapse

        self.W = W_next
        self.M = max(self.M, self.W)

        # EZ external reward (main economic objective)
        power = 1.0 - 1.0 / self.psi
        C_safe = max(C_t, 1e-8)
        r_ext = (1.0 - self.beta) * (C_safe ** power)

        # advance time
        self.t += 1
        done = self.t >= self.T - 1

        next_state = None if done else self._state()
        info = {"W": self.W, "C": C_t}

        return next_state, r_ext, done, info