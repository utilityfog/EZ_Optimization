import torch

def ez_z_target(C_t, y_next_hat, beta: float, psi: float, gamma_risk: float):
    """
    Build the one step target for z(V_t) using the current consumption and next y_hat.

    z(V_t) = (1 - beta) * C_t^{1 - 1/psi}
             + beta * [ E_t( V_{t+1}^{1 - gamma} ) ]^{ (1 - 1/psi) / (1 - gamma) }

    In our approximation we plug in y_next_hat for V_{t+1}^{1-gamma}.

    Inputs:
      C_t         tensor [T]
      y_next_hat  tensor [T]  (critic output from next state)
    """
    power_inner = 1.0 / (1.0 - gamma_risk)
    power_outer = 1.0 - 1.0 / psi

    term1 = (1.0 - beta) * torch.pow(torch.clamp(C_t, min=1e-8), power_outer)
    term2 = beta * torch.pow(torch.clamp(y_next_hat, min=1e-8), power_inner * power_outer)
    return term1 + term2