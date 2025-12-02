class Config:
    # EZ parameters
    beta = 0.96            # time preference
    psi = 1.5              # elasticity of intertemporal substitution
    gamma_risk = 5.0       # risk aversion for Epstein Zin recursion

    # fractional differencing used only for returns
    frac_d = 0.4
    frac_max_lag = 512
    frac_tol = 1e-4

    # PPO parameters
    gamma = 0.99           # discount factor used inside GAE
    gae_lambda = 0.95      # GAE smoothing
    clip_ratio = 0.2       # PPO clipping parameter
    ppo_epochs = 10
    batch_size = 128
    lr = 3e-4              # Adam learning rate

    # value loss, entropy, curiosity weights
    c_v = 1.0              # value-loss weight
    beta_ent = 1e-3        # entropy weight for Gaussian policy
    c_icm = 0.5            # curiosity weight (ICM)
    eta_icm = 1e-3         # scale for intrinsic reward inside ICM

    # training
    num_episodes = 200
    start_wealth = 1.0

    # device
    device = "cpu"