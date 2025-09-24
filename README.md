# SPEC: RL INVESTOR (PPO + ICM) with **Epstein–Zin** Preferences + **Learnable Fractional Differencing**
*A self-contained, implementation-ready specification. No code included. All math, shapes, distributions, rewards, targets, and losses are explicit.*

---

## 0) PURPOSE & SCOPE
This document **replaces CRRA** with **Epstein–Zin (EZ)** recursive preferences and **adds a Learnable Fractional Differencing (FracDiff) layer** in the feature pipeline, while preserving your **PPO + ICM** training stack and environment mechanics. It is **drop-in**: policy parameterization, buffers, rollout loop, exact log-probabilities, and PPO machinery remain intact. Only the **preference aggregator** (reward/value semantics) and **feature memory module** (FracDiff) change.

**What stays the same (do not touch):**
- Time/indexing, assets, returns, risk-free, turnover/transaction cost, budget identity.
- Actor heads (consumption squashed Gaussian; risky weights Dirichlet/softmax), exact log-prob math.
- Critic backbone mechanics (but we output two EZ heads; see §5).
- ICM encoder/forward (and optional inverse) and curiosity reward wiring.
- PPO: ratio, clipping, GAE, epochs/minibatching, entropy, optimizers.
- Data split, standardization, rollout collection, buffer contents.

**What changes:**
1) **Utility/Value:** CRRA is replaced by **Epstein–Zin** with a numerically stable target in z-space (§4–§6).  
2) **Features:** Insert **Learnable FracDiff** over returns before feature construction (§3).

---

## 1) CORE DEFINITIONS (TIME, DATA, WEALTH, CONSUMPTION, BUDGET)
### 1.1 Time and assets
- Discrete time `t = 0,1,2,...,T−1` (episode length `T`).  
- Number of risky assets `n ≥ 1`.
- Market data (known up to `t` when acting):
  - Risky **gross** returns: `R[t] ∈ ℝ^n` (e.g., 1.01 = +1%).
  - Risk-free **gross** return: `Rf[t] ∈ ℝ`.

### 1.2 Wealth, consumption, normalization
- Wealth at start of step `t`: `W_t > 0`.
- Consumption fraction (action): `c_t ∈ (0,1)`; **dollar consumption** `C_t := c_t · W_t`.
- Running max wealth `M_t := max_{0≤τ≤t} W_τ`; normalized wealth `Ṽ_t := W_t / M_t ∈ (0,1]`.

### 1.3 Portfolio, turnover, transaction cost
- Risky-asset weights (action): `w_t ∈ Δ^n` (simplex, nonnegative, sum=1).  
- Implicit cash weight: `w_cash,t := 1 − ∑_i w_t[i]` (nonnegative by construction).
- Turnover: `turnover_t := ‖w_t − w_{t−1}‖_1`.  Transaction-cost coefficient: `κ ≥ 0`.
- Dollar cost: `TC_t := κ · W_t · turnover_t` (paid immediately at `t`).

### 1.4 Budget identity (wealth transition)
Let risky **excess** return `Ṙ[t+1] := R[t+1] − Rf[t+1]·1⃗`.
- Gross growth factor:
  \[
  G_{t+1} := (1 − c_t)\,\big( Rf[t+1] + w_t^{\top} Ṙ[t+1] \big) \;-\; κ \, \|w_t − w_{t−1}\|_1.
  \]
- Next wealth: `W_{t+1} := W_t · G_{t+1}`. Safety floors may clip `G_{t+1}` to `ε_g>0`.

---

## 2) OBSERVATIONS, FEATURES, STATE (BASELINE PIPELINE)
### 2.1 Observables at time t
- `W_t`, `w_{t−1}`, and a causal feature vector `x_t ∈ ℝ^d` built **only** from data ≤ `t`.
- Standardize `x_t` via train-set `(μ,σ)` to `ẋ_t` (store `μ,σ` from training only).

### 2.2 State to networks
- **State:** `s_t := concat( Ṽ_t, ẋ_t, w_{t−1} ) ∈ ℝ^{1+d+n}` (fixed order).

---

## 3) **Learnable Fractional Differencing** (returns-domain feature module)
### 3.1 Goal & parameter
Learn a memory depth `d_target ∈ [d_min, d_max]` (e.g., `[0,1]`) that controls the fractional differencing of returns to **capture long memory** while promoting **stationarity**.

### 3.2 Placement in pipeline
- Input raw **log-returns** per asset: `r_t ∈ ℝ^n` (or windows).  
- Apply a FracDiff operator with effective exponent `d_eff`:
  - **Mode “direct”**: apply `(1 − L)^{d_target}` to returns.  
  - **Mode “price_equiv”**: apply `(1 − L)^{d_target − 1}` to returns (equivalent to price fracdiff of `d_target` without reconstructing prices).
- Truncate the kernel to length `K` (auto-chosen from `d_eff` and a tolerance). Outputs lose the first `K` steps.

### 3.3 State augmentation & alignment
- Build your usual statistics **from** the FD output (lags, MAs, vol, PCA, cross-sectional transforms).  
- **Shift** all time-aligned targets by `K` (drop first `K` steps) so shapes match.  
- Optionally append `stop_grad(d_target)` and `K` as scalar features so policy/critic can adapt to memory depth.

### 3.4 Regularization & constraints
- Keep `d_target` within bounds via a sigmoid reparameterization.  
- Add a small L2 penalty if `d_target` sticks to the bounds.  
- Optional “whiteness” regularizer: penalize low-lag autocorr of FD residuals to avoid over-memory.

> **Everything backpropagates end-to-end** because kernel weights are differentiable functions of `d_eff`.

---

## 4) **Epstein–Zin** Preferences (replace CRRA)
Let `β∈(0,1)` be the subjective discount, `γ>0` risk aversion, `ψ>0` intertemporal elasticity (EIS). Define transforms:
- \( z(V) := V^{\,1-\frac{1}{\psi}} \)   (EIS/consumption space)  
- \( y(V) := V^{\,1-\gamma} \)           (risk space)

### 4.1 EZ aggregator (Kreps–Porteus form)
For lifetime utility \(V_t\) and consumption \(C_t\):
\[
V_t \;=\; \Big[(1-\beta)\, C_t^{\,1-\frac{1}{\psi}} \;+\; \beta \,\big( \mathbb{E}_t [ V_{t+1}^{\,1-\gamma} ] \big)^{\frac{1-\frac{1}{\psi}}{\,1-\gamma\,}} \Big]^{\frac{1}{\,1-\frac{1}{\psi}\,}}.
\]

### 4.2 Practical RL parameterization (stable targets)
We **train in z-space** with a two-head critic predicting \(\hat z_t \approx z(V_t)\) and \(\hat y_t \approx y(V_t)\).  
- **External (shaped) reward:** \( r_t^{ext} := (1-\beta)\, C_t^{\,1-\frac{1}{\psi}} \).  
- **One-step bootstrap target for z:**  
\[
T^{(z)}_t \;:=\; (1-\beta)\, C_t^{\,1-\frac{1}{\psi}} \;+\; \beta \,\Big(\hat y_{t+1}\Big)^{\frac{1-\frac{1}{\psi}}{\,1-\gamma\,}}.
\]
- **Value loss:** \( L_{value} := \tfrac{1}{2}\,(\hat z_t - T^{(z)}_t)^2 \).  
- Optional **consistency** regularizer: encourage \( \hat y_t \approx (\hat z_t)^{\frac{1-\gamma}{1-\frac{1}{\psi}}}\) with a small weight.

> **Degeneracies:** ψ→1 approaches additive/separable (log-like); γ→1 reduces risk curvature; recipe reduces toward CRRA smoothly.

---

## 5) ACTOR & CRITIC (Z-functions, distributions, exact log-probs)
### 5.1 Actor `f_θ`
**Input:** `s_t ∈ ℝ^{1+d+n}` → shared backbone → three heads:
- Consumption pre-activation mean `μ_c := z_c ∈ ℝ` and log-std `ℓ_c ∈ ℝ`; set `σ_c := softplus(ℓ_c)+σ_floor`.  
  - Sample pre-squash `y_c ~ Normal(μ_c, σ_c^2)`, then **squash**: `c_t := sigmoid(y_c) ∈ (0,1)`.
- Risky weights logits `z_w ∈ ℝ^n`; **Dirichlet** concentration `α := softplus(z_w) + ε_dir` → sample `w_t ~ Dir(α)`.
- Deterministic eval: `c_t := sigmoid(μ_c)`, `w_t := α / ∑α`.

### 5.2 Exact log-probabilities (for PPO ratio)
- For consumption (squashed Gaussian): with `y_c = logit(c_t)` and Jacobian `|dy/dc| = 1/(c_t (1−c_t))`:
  \[
  \log p(c_t|s_t) \;=\; \log \mathcal{N}(y_c; μ_c, σ_c^2) \;-\; \log( c_t(1-c_t) ).
  \]
- For weights (Dirichlet):
  \[
  \log p(w_t|s_t) \;=\; \log \Gamma(\textstyle\sum_i α_i) - \sum_i \log \Gamma(α_i) + \sum_i (α_i-1)\log w_t[i].
  \]
- Joint: `log π_θ(a_t|s_t) = log p(c_t|s_t) + log p(w_t|s_t)` (store at collection).

### 5.3 Critic `g_ψ` (**two heads for EZ**)
- Input: `s_t`. Shared backbone → two scalar heads: `\hat z_t` and `\hat y_t`.  
- Only `\hat z_t` is used in advantages/GAE; `\hat y_t` feeds the z-target via §4.2.

---

## 6) ENVIRONMENT STEP (FULL SEQUENCE)
At each time `t`:
1. Build state `s_t = concat(Ṽ_t, ẋ_t, w_{t−1})` (after FD alignment, see §3.3).
2. Actor forward: get `μ_c, σ_c, α` → sample `c_t`, `w_t`; compute and **store** `log π_{θ_old}(a_t|s_t)` exactly (§5.2).
3. Consumption: `C_t := c_t · W_t`.
4. Observe `R[t+1], Rf[t+1]`; turnover `‖w_t−w_{t−1}‖_1`; cost `TC_t := κ W_t ‖·‖_1`.
5. Wealth update via §1.4 → `W_{t+1}`; update max `M_{t+1}`.
6. Build next features `x_{t+1}` (post-FracDiff), standardize `→ ẋ_{t+1}`.
7. Next state: `s_{t+1} := concat(W_{t+1}/M_{t+1}, ẋ_{t+1}, w_t)`.

---

## 7) REWARDS (EXTERNAL EZ FLOW, INTRINSIC ICM)
### 7.1 External reward (EZ flow term in z-space)
\[
r_t^{ext} := (1-\beta)\, C_t^{\,1-\frac{1}{\psi}}.
\]

### 7.2 Intrinsic curiosity reward (ICM)
Encoders and models (as in baseline):
- State encoder `φ(s) ∈ ℝ^m`; forward model `f(φ(s_t), ψ(a_t)) → \hat φ_{t+1}`; (optional) inverse model for stability.
- Action embedding `ψ(a_t) := concat( logit(c_t), w_t )`.
- Intrinsic reward:
\[
r_t^{int} := \eta \, \|\, φ(s_{t+1}) - \hat φ_{t+1} \,\|_2^2.
\]
- ICM losses: `L_fwd = ‖·‖^2`, `L_inv = −[ \log \mathcal{N}(y_c; μ̂_c, σ̂_c^2) + \log Dir(w_t; α̂) ]` (optional), `L_ICM = L_fwd + λ_{inv} L_{inv}`.

### 7.3 Total reward
`r_t := r_t^{ext} + r_t^{int}` (what enters the advantage computation).

---

## 8) ADVANTAGES, TARGETS, AND LOSSES (EZ version)
### 8.1 EZ TD residual and GAE (in z-space)
- Build bootstrap target for z using next-state head:
\[
T^{(z)}_t \;=\; (1-\beta)\, C_t^{\,1-\frac{1}{\psi}} \;+\; \beta \,\Big(\hat y_{t+1}\Big)^{\frac{1-\frac{1}{\psi}}{\,1-\gamma\,}}.
\]
- Define TD residual:
\[
\delta_t^{EZ} \;:=\; r_t \;+\; \beta\big(T^{(z)}_t - r_t^{ext}\big) \;-\; \hat z_t.
\]
  Intuition: `T^{(z)}` already contains the β-weighted continuation; `r_t^{ext}` is the immediate z-flow.  
- Compute **GAE(λ)** on `δ_t^{EZ}` exactly as baseline GAE (backward recursion).  
- Advantages are normalized per batch and used in PPO unchanged.

### 8.2 Losses
- **Policy (clipped PPO):** exactly baseline with advantages from §8.1.  
- **Value (z-head):** \( L_{value} = \tfrac{1}{2}(\hat z_t - T^{(z)}_t)^2 \).  
- **Entropy:** unchanged (Gaussian + Dirichlet).  
- **ICM:** unchanged.  
- **Total:**  
\[
L_{total} \;=\; L_{PPO} \;+\; c_v L_{value} \;+\; \beta_{ent} L_{ent} \;+\; c_{icm} L_{ICM}.
\]

---

## 9) TRAINING PROCEDURE (COLLECT → TARGETS → PPO)
### 9.1 Hyperparameters (additions/changes)
- **EZ:** choose `γ ∈ {5,10}`, `ψ ∈ {0.5,1.0,1.5}`, `β ∈ [0.95,0.999]`.
- **FracDiff:** `d_target` init 0.3–0.5 within `[0,1]`, tolerance `1e−4`, `K_max ∈ [1024,4096]` (match horizon).
- **RL:** keep PPO λ, clip, epochs, minibatch, lrs same initially.

### 9.2 Rollout collection (unchanged mechanics)
- Collect tuples `(s_t, a_t=(c_t,w_t), r_t, s_{t+1}, log π_{θ_old})` where `r_t` includes EZ flow + curiosity.  
- Align time by dropping first `K` steps due to FracDiff.

### 9.3 Target building & PPO update
- For each step, compute `T^{(z)}_t`, `δ_t^{EZ}`, GAE, and z-value loss.  
- Recompute current `log π_θ` exactly (§5.2); perform clipped PPO with entropy and ICM losses.  
- After epochs, set `θ_old ← θ`.

### 9.4 Evaluation (deterministic)
- Use `c_t := sigmoid(μ_c)`, `w_t := α/∑α`.  
- Recover EZ value via inverse transform for reporting: \( \hat V_t = \hat z_t^{\,1/(1-\frac{1}{\psi})} \).  
- Report PnL, CAGR, MDD, Calmar, turnover, and \( \hat V_0 \).

---

## 10) DIAGNOSTICS, CHECKS, & ABALATIONS
- **EZ sanity:** as ψ→1 or γ→1, curves and training behavior should smoothly approach separable/CRRA.  
- **Scale hygiene:** track `\hat z_t, \hat y_t` magnitudes; clamp/normalize if exploding.  
- **FracDiff:** monitor learned `d_target` trajectory; inspect ACF/PACF of FD outputs; avoid non-stationary drift.  
- **Alignment:** verify all post-FD tensors drop the first `K` steps; shapes of policy/value/ICM batches match.  
- **Ablations:** (i) turn off FracDiff (identity) to test EZ alone; (ii) ψ grid with fixed γ; (iii) compare CRRA vs EZ at matched γ with ψ≈1.

---

## 11) MINIMAL MIGRATION CHECKLIST
- [ ] Expose `γ, ψ, β` in config; leave PPO hypers unchanged initially.  
- [ ] Critic: switch to **two heads** `(\hat z, \hat y)`; keep shared backbone.  
- [ ] Reward pipe: compute `r_t^{ext}=(1−β)C^{1−1/ψ}`; add curiosity as before → `r_t`.  
- [ ] Targets: build `T^{(z)}` with next-state `\hat y`; compute `δ^{EZ}` and GAE in z-space.  
- [ ] Insert **Learnable FracDiff** before feature builder; shift by `K`.  
- [ ] Log `d_target`, `K`, ACF diagnostics, and `\hat z,\hat y` summaries.

---

## 12) GLOSSARY
- `C_t`: dollar consumption; `W_t`: wealth; `c_t`: consumption rate.  
- `γ`: risk aversion; `ψ`: EIS; `β`: discount.  
- `V_t`: EZ lifetime utility; `z(V)=V^{1−1/ψ}`; `y(V)=V^{1−γ}`.  
- `d_target`: fracdiff memory parameter; `K`: kernel truncation length.  
- `ICM`: Intrinsic Curiosity Module; `φ`: encoder; `f`: forward model.

---

## 13) TL;DR (one-screen summary)
- **Objective change:** CRRA → **Epstein–Zin** with a two-head critic (`\hat z, \hat y`), shaped external reward \( (1−β)C^{1−1/ψ} \), and a one-step z-target using `\hat y_{t+1}`.  
- **Feature change:** Insert **Learnable FracDiff** over returns; align time by kernel length `K`.  
- **PPO/ICM:** identical machinery; only the value target and reward semantics change.  
- **Eval:** deterministic actions; report \( \hat V_0 \) via inverse transform and standard trading metrics.
