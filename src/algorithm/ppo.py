import torch
import torch.nn.functional as F

def compute_gae(rewards, values, next_value, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros(T)
    last_adv = 0
    for t in reversed(range(T)):
        td = rewards[t] + gamma*(next_value if t==T-1 else values[t+1]) - values[t]
        advantages[t] = last_adv = td + gamma*lam*last_adv
    returns = advantages + values
    return advantages, returns

def ppo_update(model, optimizer, batch, clip=0.2, vf_coeff=0.5, ent_coeff=0.01):
    states, actions, old_logp, adv, returns, z_target = batch

    mu, std, z_hat, y_hat = model(states)
    eps = (actions - mu).detach() / (std + 1e-8)
    y = mu + eps * std
    c = torch.sigmoid(y)

    # recompute log prob
    log_norm = -0.5*((y-mu)/std)**2 - torch.log(std) - 0.5*np.log(2*np.pi)
    log_det = torch.log(c) + torch.log(1-c)
    logp = log_norm - log_det

    ratio = torch.exp(logp - old_logp)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1-clip, 1+clip) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = 0.5 * ((z_hat - z_target)**2).mean()

    entropy = -(logp).mean()

    loss = policy_loss + vf_coeff*value_loss - ent_coeff*entropy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return dict(policy_loss=policy_loss.item(),
                value_loss=value_loss.item(),
                entropy=entropy.item())