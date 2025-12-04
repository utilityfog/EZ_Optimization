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
        build_processed(
            frac_d=cfg.frac_d,
            max_lag=cfg.frac_max_lag,
            tol=cfg.frac_tol,
        )

    features = np.load(features_path)
    returns = np.load(returns_path)

    # Check for non-finite values
    bad_feat = ~np.isfinite(features)
    bad_ret = ~np.isfinite(returns)

    if bad_feat.any() or bad_ret.any():
        print("Non-finite values detected in processed data")

        # Option A - drop any rows with non-finite entries (more principled)
        feat_row_mask = np.isfinite(features).all(axis=1)
        ret_row_mask = np.isfinite(returns)
        joint_mask = feat_row_mask & ret_row_mask

        print("  keeping", joint_mask.sum(), "rows out of", joint_mask.size)

        features = features[joint_mask]
        returns = returns[joint_mask]

        # Optional: assert nothing bad left
        assert np.isfinite(features).all()
        assert np.isfinite(returns).all()

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
        window_len=cfg.fd_window,
    )

    dummy_state = env.reset()
    state_dim = dummy_state.shape[0]

    model = ActorCriticEZ(
        state_dim=state_dim,
        hidden_dim=128,
        use_fracdiff=cfg.use_learnable_fracdiff,
        fracdiff_max_lag=cfg.fracdiff_max_lag,
        fracdiff_init_d=cfg.fracdiff_init_d,
        window_len=cfg.fd_window,
    ).to(device)
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
            if np.isnan(r_ext):
                print(">>> NAN r_ext FROM ENV AT STEP", step_idx, "C=", info["C"], "W=", env.W)
                raise SystemExit
            
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
        
        # DEBUG: inspect states
        if not torch.isfinite(states).all():
            print("Non-finite states detected after rollout")
            nan_mask = torch.isnan(states)
            inf_mask = torch.isinf(states)

            print("  any NaN:", nan_mask.any().item())
            print("  any Inf:", inf_mask.any().item())

            bad_rows = nan_mask.any(dim=1) | inf_mask.any(dim=1)
            bad_idx = bad_rows.nonzero(as_tuple=False).squeeze()

            print("  timesteps with bad state rows:", bad_idx)
            if bad_idx.numel() > 0:
                i0 = bad_idx[0].item()
                print("  example bad state row (index", i0, "):")
                print(states[i0])

            raise RuntimeError("Non-finite states coming from env/preprocessing")

        # build y_next_hat by shifting
        y_next = torch.cat([y_hats[1:], y_hats[-1:].clone()], dim=0)

        z_targets = ez_z_target(
            C_t=C_arr,
            y_next_hat=y_next,
            beta=cfg.beta,
            psi=cfg.psi,
            gamma_risk=cfg.gamma_risk,
        )

        advantages, _ = compute_gae(
            rewards=rewards, values=z_hats, gamma=cfg.gamma, lam=cfg.gae_lambda
        )
        
        # print("CHECK: rewards min/max:", rewards.min().item(), rewards.max().item())
        # print("CHECK: C_arr min/max:", C_arr.min().item(), C_arr.max().item())
        # print("CHECK: z_hats min/max:", z_hats.min().item(), z_hats.max().item())
        # print("CHECK: y_next min/max:", y_next.min().item(), y_next.max().item())
        # print("CHECK: z_targets min/max:", z_targets.min().item(), z_targets.max().item())
        # print("CHECK: advantages min/max:", advantages.min().item(), advantages.max().item())


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
        
        # After training, save model weights for evaluation
        save_path = os.path.join(PROC_DIR, "actor_critic_ez.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved trained ActorCriticEZ to {save_path}")


if __name__ == "__main__":
    main()