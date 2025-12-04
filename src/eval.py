import os
import numpy as np
import torch

from .config import Config
from .data_preprocessing import PROC_DIR
from .env import EZSingleAssetEnv
from .model import ActorCriticEZ


def load_splits():
    """
    Load train and test splits created by build_processed.
    """
    train_features = np.load(os.path.join(PROC_DIR, "features.npy"))
    train_returns = np.load(os.path.join(PROC_DIR, "returns.npy"))

    test_features_path = os.path.join(PROC_DIR, "features_test.npy")
    test_returns_path = os.path.join(PROC_DIR, "returns_test.npy")

    if not (os.path.exists(test_features_path) and os.path.exists(test_returns_path)):
        raise FileNotFoundError(
            "Test split not found. Run data_preprocessing.build_processed() first."
        )

    test_features = np.load(test_features_path)
    test_returns = np.load(test_returns_path)

    return (train_features, train_returns), (test_features, test_returns)


def make_model(cfg: Config, state_dim: int, device: torch.device) -> ActorCriticEZ:
    model = ActorCriticEZ(
        state_dim=state_dim,
        hidden_dim=128,
        use_fracdiff=cfg.use_learnable_fracdiff,
        fracdiff_max_lag=cfg.fracdiff_max_lag,
        fracdiff_init_d=cfg.fracdiff_init_d,
        window_len=cfg.fd_window,
    ).to(device)
    return model


def backtest_deterministic(cfg: Config, model_path: str):
    device = torch.device(cfg.device)

    (_, _), (features_test, returns_test) = load_splits()

    # Build env on test split
    env = EZSingleAssetEnv(
        returns=returns_test,
        features=features_test,
        beta=cfg.beta,
        psi=cfg.psi,
        start_wealth=cfg.start_wealth,
        window_len=cfg.fd_window,
    )

    # Infer state dimension from env
    state_np = env.reset()
    state_dim = state_np.shape[0]

    # Build model and load weights
    model = make_model(cfg, state_dim, device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Roll through test set deterministically
    wealth_path = [env.W]
    cons_path = []
    z_path = []
    y_path = []

    state = torch.tensor(state_np, dtype=torch.float32, device=device)
    done = False

    with torch.no_grad():
        while not done:
            if state.dim() == 1:
                state_in = state.unsqueeze(0)
            else:
                state_in = state

            # Deterministic action: c_t = sigmoid(mu_c(s_t))
            mu, std, z_hat, y_hat = model.forward(state_in)
            c_det = torch.sigmoid(mu).squeeze(-1)
            c_scalar = float(c_det.cpu().item())

            next_state_np, r_ext, done, info = env.step(c_scalar)

            wealth_path.append(env.W)
            cons_path.append(info["C"])
            z_path.append(float(z_hat.squeeze(-1).cpu().item()))
            y_path.append(float(y_hat.squeeze(-1).cpu().item()))

            if not done:
                state = torch.tensor(
                    next_state_np, dtype=torch.float32, device=device
                )

    wealth = np.array(wealth_path, dtype=np.float64)
    cons = np.array(cons_path, dtype=np.float64)
    z_arr = np.array(z_path, dtype=np.float64)

    # Basic trading metrics
    initial_wealth = wealth[0]
    final_wealth = wealth[-1]
    pnl = final_wealth / initial_wealth - 1.0

    n_steps = returns_test.shape[0]
    # monthly data, so convert steps to years
    years = n_steps / 12.0 if n_steps > 0 else np.nan
    if years > 0:
        cagr = (final_wealth / initial_wealth) ** (1.0 / years) - 1.0
    else:
        cagr = np.nan

    running_max = np.maximum.accumulate(wealth)
    drawdowns = wealth / running_max - 1.0
    max_dd = float(drawdowns.min())
    if max_dd < 0:
        calmar = cagr / abs(max_dd)
    else:
        calmar = np.nan

    # Recover EZ value at t=0 from z_hat
    psi = cfg.psi
    if z_arr.size > 0:
        z0 = z_arr[0]
        # inverse of z(V) = V^{1 - 1/psi}
        exponent = 1.0 / (1.0 - 1.0 / psi)
        V0_hat = float(z0 ** exponent)
    else:
        V0_hat = np.nan

    stats = {
        "initial_wealth": float(initial_wealth),
        "final_wealth": float(final_wealth),
        "pnl": float(pnl),
        "cagr": float(cagr),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "V0_hat": float(V0_hat),
        "n_test_steps": int(n_steps),
    }

    print("Deterministic test evaluation:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return stats


def main():
    cfg = Config()
    model_path = os.path.join(PROC_DIR, "actor_critic_ez.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Run training first."
        )
    backtest_deterministic(cfg, model_path)


if __name__ == "__main__":
    main()