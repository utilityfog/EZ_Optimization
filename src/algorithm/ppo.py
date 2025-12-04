import math
import torch
import torch.nn.functional as F
from torch.distributions import Normal

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Standard GAE using rewards and value function.

    This is kept for generic use, but for the EZ setting we
    typically prefer compute_gae_from_deltas() with EZ TD residuals.
    """
    T = rewards.size(0)
    advantages = torch.zeros(T, device=rewards.device)
    last_adv = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        last_adv = delta + gamma * lam * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages, returns


def compute_gae_from_deltas(deltas, gamma=0.99, lam=0.95):
    """
    GAE variant that works directly on TD residuals (deltas).

    Given EZ TD residuals delta_t^{EZ}, we define:

        A_t = delta_t^{EZ}
              + (gamma * lam) delta_{t+1}^{EZ}
              + (gamma * lam)^2 delta_{t+2}^{EZ}
              + ...

    Inputs:
      deltas: tensor [T] of TD residuals
      gamma:  scalar discount (typically beta for EZ)
      lam:    GAE parameter in [0,1]

    Returns:
      advantages: tensor [T]
    """
    T = deltas.size(0)
    advantages = torch.zeros(T, device=deltas.device)
    last_adv = 0.0

    for t in reversed(range(T)):
        last_adv = deltas[t] + gamma * lam * last_adv
        advantages[t] = last_adv

    return advantages


def ppo_update(
    model,
    optimizer,
    states,
    actions,
    old_logp,
    advantages,
    z_targets,
    clip_ratio=0.2,
    vf_coeff=0.5,
    ent_coeff=0.0,
    epochs=10,
    batch_size=64,
):
    """
    Perform PPO updates on a batch of data.

    Inputs:
      model:      ActorCriticEZ module
      optimizer:  torch optimizer
      states:     [N, state_dim]
      actions:    [N], consumption rates c_t in (0,1)
      old_logp:   [N], behavior log probabilities stored at rollout
      advantages: [N], precomputed (and optionally normalized)
      z_targets:  [N], EZ z-targets T^{(z)}_t
    """
    N = states.size(0)
    # normalize advantages once per update
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        perm = torch.randperm(N)
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            s_b = states[idx]
            a_b = actions[idx]
            old_logp_b = old_logp[idx]
            adv_b = advantages[idx]
            z_targ_b = z_targets[idx]
            
            # Check data coming in
            if not torch.isfinite(s_b).all():
                print("Non-finite states in batch")
                raise RuntimeError("Non-finite states")

            if not torch.isfinite(a_b).all():
                print("Non-finite actions in batch")
                raise RuntimeError("Non-finite actions")

            if not torch.isfinite(old_logp_b).all():
                print("Non-finite old_logp in batch")
                print("old_logp_b:", old_logp_b)
                raise RuntimeError("Non-finite old_logp")

            if not torch.isfinite(adv_b).all():
                print("Non-finite advantages in batch")
                print("adv_b:", adv_b)
                raise RuntimeError("Non-finite advantages")

            if not torch.isfinite(z_targ_b).all():
                print("Non-finite z_targets in batch")
                print("z_targ_b:", z_targ_b)
                raise RuntimeError("Non-finite z_targets")

            mu, std, z_hat, y_hat = model.forward(s_b)
            
            # Hard safety on mu and std before anything else
            if not torch.isfinite(mu).all():
                print("Non-finite mu from model.forward")
                # print("mu min/max:", torch.nanmin(mu), torch.nanmax(mu))
                raise RuntimeError("NaN or Inf in mu")

            if not torch.isfinite(std).all():
                print("Non-finite std from model.forward (before clamp)")
                # print("std min/max:", torch.nanmin(std), torch.nanmax(std))
                raise RuntimeError("NaN or Inf in std before clamp")

            # clamp std directly to avoid degenerate sigmas
            std = torch.clamp(std, 1e-3, 5.0)
            mu = torch.clamp(mu, -20.0, 20.0)

            # reconstruct pre-sigmoid y corresponding to given actions a_b
            a_clamped = torch.clamp(a_b, 1e-6, 1.0 - 1e-6)
            y = torch.log(a_clamped) - torch.log(1.0 - a_clamped)
            # optional clamp, since y outside [-20,20] is effectively saturated anyway
            y = torch.clamp(y, -20.0, 20.0)

            # Stable Gaussian log-prob in y-space
            dist = Normal(mu, std)
            log_y = dist.log_prob(y)

            # log|det dσ^{-1}/dy| = -log(a(1-a)), so we subtract log_det
            log_det = torch.log(a_clamped) + torch.log(1.0 - a_clamped)

            logp = (log_y - log_det).squeeze(-1)

            # debug guards
            if torch.isnan(logp).any():
                print("NaN in logp; inspecting pieces...")
                print("y finite:", torch.isfinite(y).all())
                print("mu finite:", torch.isfinite(mu).all())
                print("std finite:", torch.isfinite(std).all())
                print("log_y finite:", torch.isfinite(log_y).all())
                print("log_det finite:", torch.isfinite(log_det).all())
                raise RuntimeError("NaN in logp")
            
            # debug guards (optional but very useful while developing)
            if torch.isnan(logp).any():
                print("NaN in logp")
            if torch.isnan(z_hat).any():
                print("NaN in z_hat")
            if torch.isnan(adv_b).any():
                print("NaN in adv_b")


            # PPO clipped surrogate objective
            ratio = torch.exp(logp - old_logp_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            # value loss on z-head
            value_loss = 0.5 * F.mse_loss(z_hat, z_targ_b)

            # entropy of the pre-squash Normal
            # H = 0.5 * log(2πeσ^2)
            H_c = 0.5 * torch.log(2.0 * math.pi * math.e * std**2)
            entropy = H_c.mean()

            loss = policy_loss + vf_coeff * value_loss - ent_coeff * entropy
            
            if not torch.isfinite(loss):
                print("Non-finite loss detected")
                print("policy_loss:", policy_loss.item())
                print("value_loss:", value_loss.item())
                print("entropy:", entropy.item())
                # print("ratio min/max:", torch.nanmin(ratio), torch.nanmax(ratio))
                raise RuntimeError("Non-finite loss")

            optimizer.zero_grad()
            loss.backward()
            
            # Check grads before clipping
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    print("Non-finite grad in", name)
                    # print("grad min/max:", torch.nanmin(p.grad), torch.nanmax(p.grad))
                    raise RuntimeError("Non-finite gradient")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Check params after step
            for name, p in model.named_parameters():
                if not torch.isfinite(p).all():
                    print("Non-finite parameter after optimizer.step in", name)
                    # print("param min/max:", torch.nanmin(p.data), torch.nanmax(p.data))
                    raise RuntimeError("Non-finite parameter after step")

    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
    }