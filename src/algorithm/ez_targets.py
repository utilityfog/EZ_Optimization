import torch

def ez_z_target(C_t, y_next_hat, beta: float, psi: float, gamma_risk: float):
    """
    Build the one-step EZ target for z(V_t) using C_t and next-state y_hat.

    The theoretical form:

        T^{(z)}_t
          = (1 - beta) * C_t^{1 - 1/psi}
            + beta * [ E_t( V_{t+1}^{1 - gamma} ) ]^{ (1 - 1/psi)/(1 - gamma) }

    In our approximation we plug in y_next_hat for V_{t+1}^{1 - gamma}.

    Inputs:
      C_t        tensor [...], consumption level
      y_next_hat tensor [...], critic y output from next state (approx V_{t+1}^{1-gamma})

    Returns:
      T_z        tensor [...], one-step bootstrap target for z-head
    """
    # outer exponent on C_t term
    power_outer = 1.0 - 1.0 / psi
    # exponent for y_next_hat
    power_inner = power_outer / (1.0 - gamma_risk)

    C_safe = torch.clamp(C_t, min=1e-8)
    y_safe = torch.clamp(y_next_hat, min=1e-8)

    term1 = (1.0 - beta) * torch.pow(C_safe, power_outer)
    term2 = beta * torch.pow(y_safe, power_inner)
    return term1 + term2


def ez_td_residual(
    r_ext_t,
    r_int_t,
    z_hat_t,
    C_t,
    y_next_hat,
    beta: float,
    psi: float,
    gamma_risk: float,
):
    """
    Compute the EZ TD residual delta_t^{EZ}:

        delta_t^{EZ}
          = r_t + beta * (T^{(z)}_t - r_t^{ext}) - z_hat_t

    where:
      r_t = r_ext_t + r_int_t
      T^{(z)}_t is built from C_t and y_next_hat via ez_z_target().

    Inputs:
      r_ext_t     tensor [...], external EZ reward
      r_int_t     tensor [...], intrinsic reward
      z_hat_t     tensor [...], critic z output at time t
      C_t         tensor [...], consumption at time t
      y_next_hat  tensor [...], critic y output at t+1
    """
    T_z = ez_z_target(C_t, y_next_hat, beta, psi, gamma_risk)
    r_t = r_ext_t + r_int_t
    delta = r_t + beta * (T_z - r_ext_t) - z_hat_t
    return delta