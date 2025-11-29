import torch

def ez_target_z(C_t, y_next_hat, beta, psi, gamma):
    # power exponent: (1 - 1/psi) / (1 - gamma)
    power = (1 - 1/psi) / (1 - gamma)
    return (1 - beta) * (C_t ** (1 - 1/psi)) + beta * (y_next_hat ** power)