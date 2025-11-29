class Config:
    beta = 0.96
    psi = 1.5
    gamma = 5.0

    ppo_clip = 0.2
    gae_lambda = 0.95
    ppo_epochs = 10
    batch_size = 64
    lr = 3e-4
    horizon = 256