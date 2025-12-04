import torch

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.C = []
        self.z_hats = []
        self.y_hats = []

    def add(self, state, action, logp, reward, C_t, z_hat, y_hat):
        self.states.append(state)
        self.actions.append(action)
        self.logps.append(logp)
        self.rewards.append(reward)
        self.C.append(C_t)
        self.z_hats.append(z_hat)
        self.y_hats.append(y_hat)

    def to_tensors(self, device="mps"):
        states = torch.stack(self.states).to(device)
        actions = torch.stack(self.actions).to(device)
        logps = torch.stack(self.logps).to(device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        C = torch.tensor(self.C, dtype=torch.float32, device=device)
        z_hats = torch.stack(self.z_hats).to(device)
        y_hats = torch.stack(self.y_hats).to(device)
        return states, actions, logps, rewards, C, z_hats, y_hats