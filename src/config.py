class Config:
    # EZ parameters
    beta = 0.96
    psi = 1.5
    gamma_risk = 5.0

    # fracdiff preprocessing (still used by data_preprocessing)
    frac_d = 0.4
    frac_tol = 1e-4
    frac_max_lag = 512

    # learnable fractional differencing (inference-time)
    use_learnable_fracdiff = True
    fd_window = 12                # 12 months window in the state
    fracdiff_max_lag = 12         # kernel length in LearnableFracDiff1D
    fracdiff_init_d = 0.4

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
    start_wealth = 100.0

    device = "mps"