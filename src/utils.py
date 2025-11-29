import torch

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.z_hats = []
        self.y_hats = []

    def add(self, s, a, lp, r, z, y):
        self.states.append(s)
        self.actions.append(a)
        self.logps.append(lp)
        self.rewards.append(r)
        self.z_hats.append(z)
        self.y_hats.append(y)

    def clear(self):
        self.__init__()

    def to_tensors(self):
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.logps),
            torch.tensor(self.rewards),
            torch.stack(self.z_hats),
            torch.stack(self.y_hats),
        )