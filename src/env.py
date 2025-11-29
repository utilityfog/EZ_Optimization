import numpy as np

class EZSingleAssetEnv:
    def __init__(self, returns, beta, psi, start_wealth=1.0):
        self.returns = returns
        self.T = len(returns) - 1
        self.beta = beta
        self.psi = psi
        self.start_wealth = start_wealth

    def reset(self):
        self.t = 0
        self.W = float(self.start_wealth)
        self.M = self.W
        return self._build_state(self.t)

    def _features(self, t, k=10):
        start = max(0, t-k)
        pad = k - (t-start)
        window = self.returns[start:t]
        if pad > 0:
            window = np.concatenate([np.ones(pad), window])
        return window.astype(np.float32)

    def _build_state(self, t):
        W_tilde = self.W / self.M
        x_t = self._features(t)
        return np.concatenate([[W_tilde], x_t]).astype(np.float32)

    def step(self, c_t):
        C_t = c_t * self.W

        R_next = self.returns[self.t + 1]
        W_next = (1 - c_t) * self.W * R_next
        self.W = max(W_next, 1e-8)
        self.M = max(self.M, self.W)

        r_ext = (1 - self.beta) * (C_t ** (1 - 1/self.psi))

        self.t += 1
        done = self.t >= self.T
        next_state = self._build_state(self.t) if not done else None

        return next_state, r_ext, done, dict(W=self.W, C=C_t)