import numpy as np

class EZSingleAssetEnv:
    """
    Single risky asset environment for Epstein-Zin consumption timing.

    Action: c_t, the fraction of wealth consumed in (0,1).
    Wealth dynamics:
        C_t = c_t * W_t
        W_{t+1} = (W_t - C_t) * R_t
    External reward:
        r_ext_t = (1 - beta) * C_t^{1 - 1/psi}

    All remaining wealth is automatically invested in the risky asset.
    There are no portfolio weights, turnover costs, or transaction costs.

    State layout at time t:

        [ W_norm,
          features_t (d dims),
          simple_return_window (fd_window dims) ]
    """

    def __init__(self, returns, features, beta, psi, start_wealth=1.0, window_len=12, k_terminal=0.0):
        """
        returns:  array [T] of gross returns R_t (e.g. 1 + r_t)
        features: array [T, d] of feature vectors
        window_len: length of the simple-return window included in state
        """
        assert len(returns) == len(features)

        self.returns = returns.astype(np.float32)         # gross
        self.simple_returns = (returns.astype(np.float32) - 1.0)  # r_t
        self.features = features.astype(np.float32)
        self.T = len(returns)

        self.beta = float(beta)
        self.psi = float(psi)
        self.start_wealth = float(start_wealth)
        self.window_len = int(window_len)
        self.k_terminal = float(k_terminal)

        self.reset()

    def reset(self):
        self.t = 0
        self.W = self.start_wealth
        self.M = self.W     # running max for normalized wealth
        return self._state()

    def _return_window(self):
        """
        Build a fixed-length window of *past* simple returns at time t.

        We want 1-month delayed feedback:
        - At time t, the agent should only see returns up to t-1.
        - The return for this step (index t) is used in the wealth update
            and only becomes observable at t+1.

        If there is not enough history, left-pad with zeros.
        """
        K = self.window_len
        t = self.t

        # We want past K returns, excluding the current step's return.
        end = t                   # exclusive
        start = max(0, end - K)   # inclusive

        window = self.simple_returns[start:end]  # shape <= K
        window = window.astype(np.float32)

        if len(window) < K:
            pad_len = K - len(window)
            pad = np.zeros(pad_len, dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)

        return window  # shape [K]


    def _state(self):
        """
        State = [normalized wealth, features_t, return_window_t]
        normalized wealth = W_t / max past wealth.
        """
        W_norm = self.W / self.M if self.M > 0 else 1.0
        x_t = self.features[self.t]              # [d]
        r_win = self._return_window()            # [window_len]

        return np.concatenate([[W_norm], x_t, r_win]).astype(np.float32)

    def step(self, c_t):
        """
        c_t in (0,1). Clip to avoid undefined logit values.
        Computes C_t, W_{t+1}, and EZ external reward.
        """

        c_t = float(np.clip(c_t, 1e-6, 1.0 - 1e-6))

        # consumption
        C_t = c_t * self.W

        # return for this period (gross)
        R_t = float(self.returns[self.t])

        # wealth update
        W_next = (self.W - C_t) * R_t
        W_next = max(W_next, 1e-3)  # avoid numerical collapse

        self.W = W_next
        self.M = max(self.M, self.W)

        # EZ external reward (main economic objective)
        power = max(1.0 - 1.0 / self.psi, 0.0)
        C_safe = max(C_t, 1e-6)
        r_ext = (1.0 - self.beta) * (C_safe ** power)
        if not np.isfinite(r_ext):
            r_ext = 0.0
        r_ext = float(np.clip(r_ext, -1e2, 1e2))

        # advance time
        self.t += 1
        done = self.t >= self.T - 1
        
        # terminal bonus based on terminal wealth
        if done and self.k_terminal != 0.0:
            r_ext += self.k_terminal * np.log(self.W + 1e-6)

        next_state = None if done else self._state()
        info = {"W": self.W, "C": C_t}

        return next_state, r_ext, done, info