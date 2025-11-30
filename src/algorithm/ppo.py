import math
import torch
import torch.nn.functional as F

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    rewards: tensor [T]
    values: tensor [T]
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
    N = states.size(0)
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
            # reconstruct y (pre-sigmoid) that would give us c = a_b
            # solve a = sigmoid(y) -> y = log(a/(1-a))
            a_clamped = torch.clamp(a_b, 1e-6, 1.0 - 1e-6)
            y = torch.log(a_clamped) - torch.log(1.0 - a_clamped)

            log_norm = (
                -0.5 * ((y - mu) / std) ** 2
                - torch.log(std)
                - 0.5 * math.log(2.0 * math.pi)
            )
            log_det = torch.log(a_clamped) + torch.log(1.0 - a_clamped)
            logp = (log_norm - log_det).squeeze(-1)

            ratio = torch.exp(logp - old_logp_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * F.mse_loss(z_hat, z_targ_b)

            entropy = -logp.mean()
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