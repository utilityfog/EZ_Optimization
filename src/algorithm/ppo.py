import math
import torch
import torch.nn.functional as F


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

            mu, std, z_hat, y_hat = model.forward(s_b)

            # clamp std directly to avoid degenerate sigmas
            std = torch.clamp(std, 1e-3, 5.0)

            # reconstruct pre-sigmoid y corresponding to given actions a_b
            a_clamped = torch.clamp(a_b, 1e-6, 1.0 - 1e-6)
            y = torch.log(a_clamped) - torch.log(1.0 - a_clamped)

            # log probability under Gaussian, then squash via sigmoid
            log_norm = (
                -0.5 * ((y - mu) / std) ** 2
                - torch.log(std)
                - 0.5 * math.log(2.0 * math.pi)
            )
            log_det = torch.log(a_clamped) + torch.log(1.0 - a_clamped)
            logp = (log_norm - log_det).squeeze(-1)
            
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

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
    }