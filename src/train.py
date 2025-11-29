import numpy as np
import torch
from torch.optim import Adam

from data_preprocessing import load_sp500, compute_gross_returns
from env import EZSingleAssetEnv
from model import ActorCriticEZ
from algorithm.ppo import compute_gae, ppo_update
from algorithm.ez_targets import ez_target_z
from config import Config
from utils import RolloutBuffer

def main():
    cfg = Config()

    returns = np.load("data/processed/returns.npy")
    env = EZSingleAssetEnv(returns, cfg.beta, cfg.psi)

    dummy_state = env.reset()
    state_dim = len(dummy_state)

    model = ActorCriticEZ(state_dim)
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    for episode in range(2000):
        buf = RolloutBuffer()
        s = torch.tensor(env.reset()).float()

        for t in range(cfg.horizon):
            c, logp, z_hat, y_hat = model.act(s)
            c_np = c.detach().numpy().item()

            next_s_np, r_ext, done, info = env.step(c_np)
            next_s = torch.tensor(next_s_np).float() if not done else None

            buf.add(s, c, logp, r_ext, z_hat, y_hat)

            if done:
                break
            s = next_s

        states, actions, logps_old, rewards, z_hats, y_hats = buf.to_tensors()

        # y_hat_{t+1} (shifted)
        y_next = torch.cat([y_hats[1:], y_hats[-1:]], dim=0)

        # C_t recovered from s and a?
        C_t = actions.squeeze() * 1   # if you want exact C_t, store from env

        z_targets = ez_target_z(C_t, y_next, cfg.beta, cfg.psi, cfg.gamma)

        advantages, returns = compute_gae(rewards, z_hats, z_hats[-1])

        batch = (states, actions, logps_old, advantages, returns, z_targets)

        stats = ppo_update(model, optimizer, batch)

        if episode % 10 == 0:
            print(f"ep {episode} | {stats}")

if __name__ == "__main__":
    main()