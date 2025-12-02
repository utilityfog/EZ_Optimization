class Config:
    # EZ parameters
    beta = 0.96
    psi = 1.5
    gamma_risk = 5.0

    # fracdiff preprocessing
    frac_d = 0.4
    frac_tol = 1e-4
    frac_max_lag = 512

    # PPO
    gamma = 0.99
    gae_lambda = 0.95
    clip_ratio = 0.2
    vf_coeff = 0.5
    ent_coeff = 0.0
    ppo_epochs = 10
    batch_size = 128
    lr = 3e-4

    # training
    num_episodes = 200
    start_wealth = 1.0

    device = "cpu"