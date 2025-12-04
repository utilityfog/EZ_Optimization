import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam

from .config import Config
from .data_preprocessing import build_processed, PROC_DIR
from .env import EZSingleAssetEnv
from .model import ActorCriticEZ
from .algorithm.ez_targets import ez_z_target
from .algorithm.ppo import ppo_update
from .utils import RolloutBuffer
from .algorithm.ez_targets import ez_z_target, ez_td_residual
from .algorithm.ppo import compute_gae_from_deltas


def clean_features_and_returns(features: np.ndarray,
                               returns: np.ndarray,
                               split_name: str):
    """
    Make features and returns finite for a given split.

    1. Prefer to drop rows that have any non finite entries.
    2. If that would drop *all* rows, fall back to column wise imputation:
       - replace non finite feature entries with column means (or 0 if column is entirely bad)
       - replace non finite returns with 1.0 (neutral gross return)
    This guarantees:
        - no NaN or Inf in the output
        - we never end up with zero length splits.
    """
    bad_feat = ~np.isfinite(features)
    bad_ret = ~np.isfinite(returns)

    if not bad_feat.any() and not bad_ret.any():
        # Fast path: nothing to fix
        return features, returns

    # Row level mask: require every feature and the return to be finite
    feat_row_mask = np.isfinite(features).all(axis=1)
    ret_row_mask = np.isfinite(returns)
    joint_mask = feat_row_mask & ret_row_mask
    n_keep = int(joint_mask.sum())

    if n_keep > 0:
        print(f"Non-finite values detected in {split_name} split during cleaning")
        print(f"  keeping {n_keep} rows out of {joint_mask.size}")
        features_clean = features[joint_mask].copy()
        returns_clean = returns[joint_mask].copy()
        assert np.isfinite(features_clean).all()
        assert np.isfinite(returns_clean).all()
        return features_clean, returns_clean

    # Fallback: dropping would kill the entire split
    print(f"Warning: no fully finite rows in {split_name} split.")
    print("  Falling back to column-wise imputation instead of dropping everything.")

    features_clean = features.copy()
    returns_clean = returns.copy()

    # Debug: how bad is it
    nonfinite_per_col = bad_feat.sum(axis=0)
    print(f"  non-finite feature counts per column ({split_name}): {nonfinite_per_col}")
    print(f"  non-finite returns in {split_name}: {int(bad_ret.sum())}")

    # Compute column means ignoring non finite entries
    with np.errstate(invalid="ignore"):
        col_means = np.nanmean(
            np.where(np.isfinite(features_clean), features_clean, np.nan),
            axis=0,
        )

    # If a column is entirely non finite, its mean is NaN: set those to 0
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)

    # Impute features column by column
    for j in range(features_clean.shape[1]):
        col = features_clean[:, j]
        mask_bad = ~np.isfinite(col)
        if mask_bad.any():
            col[mask_bad] = col_means[j]
            features_clean[:, j] = col

    # Impute returns (gross returns, so 1.0 is "no growth")
    mask_bad_r = ~np.isfinite(returns_clean)
    if mask_bad_r.any():
        returns_clean[mask_bad_r] = 1.0

    assert np.isfinite(features_clean).all()
    assert np.isfinite(returns_clean).all()

    return features_clean, returns_clean

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

    # Use the same cleaning routine as eval, tagged as train
    features, returns = clean_features_and_returns(features, returns, split_name="train")

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
        k_terminal=cfg.k_terminal,
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
        
        actions_log = []
        wealth_log  = []

        while not done:
            c_t, logp, z_hat, y_hat = model.act(state)
            c_scalar = float(c_t.detach().cpu().item())

            next_state_np, r_ext, done, info = env.step(c_scalar)
            
            actions_log.append(c_scalar)
            wealth_log.append(info["W"])
            
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
            
        # DEBUG Plot
        plt.figure()
        plt.plot(actions_log)
        plt.title("Consumption rate c_t")
        plt.figure()
        plt.semilogy(wealth_log)
        plt.title("Wealth trajectory")
        plt.show()

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

        # bootstrap targets
        z_targets = ez_z_target(
            C_t=C_arr,
            y_next_hat=y_next,
            beta=cfg.beta,
            psi=cfg.psi,
            gamma_risk=cfg.gamma_risk,
        )
        
        # V^{1-γ}  ↔  z(V)^{(1-γ)/(1-1/ψ)}
        exp_y = (1.0 - cfg.gamma_risk) / (1.0 - 1.0 / cfg.psi)
        y_targets = torch.pow(z_targets, exp_y).clamp(min=1e-8, max=1e2)
        
        # EZ TD residuals
        r_int = torch.zeros_like(rewards) # no curiosity yet
        deltas = ez_td_residual(
            r_ext_t=rewards,
            r_int_t=r_int,
            z_hat_t=z_hats,
            C_t=C_arr,
            y_next_hat=y_next,
            beta=cfg.beta,
            psi=cfg.psi,
            gamma_risk=cfg.gamma_risk,
        )

        # GAE from deltas
        advantages = compute_gae_from_deltas(
            deltas=deltas,
            gamma=cfg.beta, # EZ discount
            lam=cfg.gae_lambda,
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
            y_targets=y_targets,
            value_scale=0.01,
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