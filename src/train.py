import os
import numpy as np
import torch
from torch.optim import Adam

from .config import Config
from .data_preprocessing import build_processed, PROC_DIR
from .env import EZSingleAssetEnv
from .model import ActorCriticEZ
from .algorithm.ez_targets import ez_z_target
from .algorithm.ppo import compute_gae, ppo_update
from .utils import RolloutBuffer

def ensure_processed(cfg: Config):
    features_path = os.path.join(PROC_DIR, "features.npy")
    returns_path = os.path.join(PROC_DIR, "returns.npy")
    if not (os.path.exists(features_path) and os.path.exists(returns_path)):
        print("Processed data not found, running preprocessing")
        build_processed(frac_d=cfg.frac_d, max_lag=cfg.frac_max_lag, tol=cfg.frac_tol)

    features = np.load(features_path)
    returns = np.load(returns_path)
    return features, returns


def main():
    cfg = Config()

    device = torch.device(cfg.device)

    features, returns = ensure_processed(cfg)

    env = EZSingleAssetEnv(
        returns=returns,
        features=features,
        beta=cfg.beta,
        psi=cfg.psi,
        start_wealth=cfg.start_wealth,
    )

    dummy_state = env.reset()
    state_dim = dummy_state.shape[0]

    model = ActorCriticEZ(state_dim=state_dim, hidden_dim=128).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    for episode in range(cfg.num_episodes):
        buf = RolloutBuffer()
        state_np = env.reset()
        state = torch.tensor(state_np, dtype=torch.float32, device=device)

        done = False
        step_idx = 0

        while not done:
            c_t, logp, z_hat, y_hat = model.act(state)
            c_scalar = float(c_t.detach().cpu().item())

            next_state_np, r_ext, done, info = env.step(c_scalar)
            C_t = info["C"]

            buf.add(
                state,
                c_t.detach(),
                logp.detach(),
                r_ext,
                C_t,
                z_hat.detach(),
                y_hat.detach(),
            )

            if not done:
                state = torch.tensor(
                    next_state_np, dtype=torch.float32, device=device
                )

            step_idx += 1

        states, actions, logps_old, rewards, C_arr, z_hats, y_hats = buf.to_tensors(
            device=device
        )

        # build y_next_hat by shifting
        y_next = torch.cat([y_hats[1:], y_hats[-1:].clone()], dim=0)

        z_targets = ez_z_target(
            C_t=C_arr,
            y_next_hat=y_next,
            beta=cfg.beta,
            psi=cfg.psi,
            gamma_risk=cfg.gamma_risk,
        )

        advantages, returns_gae = compute_gae(
            rewards=rewards, values=z_hats, gamma=cfg.gamma, lam=cfg.gae_lambda
        )

        stats = ppo_update(
            model=model,
            optimizer=optimizer,
            states=states,
            actions=actions,
            old_logp=logps_old,
            advantages=advantages,
            z_targets=z_targets,
            clip_ratio=cfg.clip_ratio,
            vf_coeff=cfg.vf_coeff,
            ent_coeff=cfg.ent_coeff,
            epochs=cfg.ppo_epochs,
            batch_size=cfg.batch_size,
        )

        avg_reward = rewards.mean().item()
        print(
            f"Episode {episode:04d}  steps={step_idx:4d}  "
            f"avg_r_ext={avg_reward:.6f}  "
            f"pol_loss={stats['policy_loss']:.4f}  "
            f"val_loss={stats['value_loss']:.4f}"
        )


if __name__ == "__main__":
    main()