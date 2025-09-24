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
1) **Utility/Value:** CRRA is replaced by **Epstein–Zin** with a numerically stable target in \(z\)-space (§4–§6).  
2) **Features:** Insert **Learnable FracDiff** over returns before feature construction (§3).

---

## 1) CORE DEFINITIONS (TIME, DATA, WEALTH, CONSUMPTION, BUDGET)
### 1.1 Time and assets
- Discrete time $\(t = 0,1,2,\ldots,T-1\)$ (episode length $\(T\)$).  
- Number of risky assets $\(n \ge 1\)$.
- Market data (known up to $\(t\)$ when acting):
  - Risky **gross** returns: $\(R[t] \in \mathbb{R}^n\)$ (e.g., $\(1.01 = +1\%\)$).
  - Risk-free **gross** return: $\(R_f[t] \in \mathbb{R}\)$.

### 1.2 Wealth, consumption, normalization
- Wealth at start of step $\(t\)$: $\(W_t > 0\)$.
- Consumption fraction (action): $\(c_t \in (0,1)\)$; **dollar consumption** $\(C_t := c_t \cdot W_t\)$.
- Running max wealth $\(M_t := \max_{0\le \tau \le t} W_\tau\)$; normalized wealth $\(\tilde W_t := W_t / M_t \in (0,1]\)$.

### 1.3 Portfolio, turnover, transaction cost
- Risky-asset weights (action): $\(w_t \in \Delta^n\)$ (simplex, nonnegative, $\(\sum_i w_t[i]=1\)$).  
- Implicit cash weight: $\(w_{\text{cash},t} := 1 - \sum_{i=1}^n w_t[i]\)$ (nonnegative by construction).
- Turnover: $ \mathrm{turnover}_t = \lVert w_t - w_{t-1} \rVert_1 $
- Turnover: $\mathrm{turnover}_t := \lVert w_t - w_{t-1} \rVert_1$.
- Turnover: \(\mathrm{turnover}_t := \lVert w_t - w_{t-1} \rVert_1\).
- Transaction-cost coefficient: $\(\kappa \ge 0\)$.
- Dollar cost: $\(\mathrm{TC}_t := \kappa \cdot W_t \cdot \mathrm{turnover}_t\)$ (paid immediately at $\(t\)$).

### 1.4 Budget identity (wealth transition)
Let risky **excess** return $\(\tilde R[t+1] := R[t+1] - R_f[t+1]\cdot \mathbf{1}\)$.
- Gross growth factor:
$$
\[
G_{t+1} := (1 - c_t)\,\big( R_f[t+1] + w_t^{\top} \tilde R[t+1] \big) \;-\; \kappa \, \lVert w_t - w_{t-1}\rVert_1.
\]
$$
- Next wealth: $\(W_{t+1} := W_t \cdot G_{t+1}\)$. Safety floors may clip $\(G_{t+1}\)$ to $\(\varepsilon_g>0\)$.

---

## 2) OBSERVATIONS, FEATURES, STATE (BASELINE PIPELINE)
### 2.1 Observables at time \(t\)
- $\(W_t\)$, $\(w_{t-1}\)$, and a causal feature vector $\(x_t \in \mathbb{R}^d\)$ built **only** from data $\(\le t\)$.
- Standardize $\(x_t\)$ via train-set $\((\mu,\sigma)\)$ to $\(\tilde x_t\)$ (store $\(\mu,\sigma\)$ from training only).

### 2.2 State to networks
- **State:** \(s_t := \mathrm{concat}\big( \tilde W_t, \tilde x_t, w_{t-1} \big) \in \mathbb{R}^{1+d+n}\) (fixed order).

---

## 3) **Learnable Fractional Differencing** (returns-domain feature module)
### 3.1 Goal & parameter
Learn a memory depth \(d_{\text{target}} \in [d_{\min}, d_{\max}]\) (e.g., \([0,1]\)) that controls the fractional differencing of returns to **capture long memory** while promoting **stationarity**.

### 3.2 Placement in pipeline
- Input raw **log-returns** per asset: \(r_t \in \mathbb{R}^n\) (or windows).  
- Apply a FracDiff operator with effective exponent \(d_{\text{eff}}\):
  - **Mode “direct”**: apply \((1 - L)^{d_{\text{target}}}\) to returns.  
  - **Mode “price\_equiv”**: apply \((1 - L)^{d_{\text{target}} - 1}\) to returns (equivalent to price fracdiff of \(d_{\text{target}}\) without reconstructing prices).
- Truncate the kernel to length \(K\) (auto-chosen from \(d_{\text{eff}}\) and a tolerance). Outputs lose the first \(K\) steps.

### 3.3 State augmentation & alignment
- Build your usual statistics **from** the FD output (lags, MAs, vol, PCA, cross-sectional transforms).  
- **Shift** all time-aligned targets by \(K\) (drop first \(K\) steps) so shapes match.  
- Optionally append \(\mathrm{stop\_grad}(d_{\text{target}})\) and \(K\) as scalar features so the policy/critic can adapt to memory depth.

### 3.4 Regularization & constraints
- Keep \(d_{\text{target}}\) within bounds via a sigmoid reparameterization.  
- Add a small L2 penalty if \(d_{\text{target}}\) sticks to the bounds.  
- Optional “whiteness” regularizer: penalize low-lag autocorrelation of FD residuals to avoid over-memory.

> **Everything backpropagates end-to-end** because kernel weights are differentiable functions of \(d_{\text{eff}}\).

---

## 4) **Epstein–Zin** Preferences (replace CRRA)
Let \(\beta\in(0,1)\) be the subjective discount, \(\gamma>0\) risk aversion, \(\psi>0\) intertemporal elasticity (EIS). Define transforms:
- \( z(V) := V^{\,1-\frac{1}{\psi}} \)   (EIS/consumption space)  
- \( y(V) := V^{\,1-\gamma} \)           (risk space)

### 4.1 EZ aggregator (Kreps–Porteus form)
For lifetime utility \(V_t\) and consumption \(C_t\):
\[
V_t \;=\; \Big[(1-\beta)\, C_t^{\,1-\frac{1}{\psi}} \;+\; \beta \,\big( \mathbb{E}_t [ V_{t+1}^{\,1-\gamma} ] \big)^{\frac{1-\frac{1}{\psi}}{\,1-\gamma\,}} \Big]^{\frac{1}{\,1-\frac{1}{\psi}\,}}.
\]

### 4.2 Practical RL parameterization (stable targets)
We **train in \(z\)-space** with a two-head critic predicting \(\hat z_t \approx z(V_t)\) and \(\hat y_t \approx y(V_t)\).  
- **External (shaped) reward:** \( r_t^{\mathrm{ext}} := (1-\beta)\, C_t^{\,1-\frac{1}{\psi}} \).  
- **One-step bootstrap target for \(z\):**  
\[
T^{(z)}_t \;:=\; (1-\beta)\, C_t^{\,1-\frac{1}{\psi}} \;+\; \beta \,\Big(\hat y_{t+1}\Big)^{\frac{1-\frac{1}{\psi}}{\,1-\gamma\,}}.
\]
- **Value loss:** \( L_{\mathrm{value}} := \tfrac{1}{2}\,\big(\hat z_t - T^{(z)}_t\big)^2 \).  
- Optional **consistency** regularizer: encourage \( \hat y_t \approx (\hat z_t)^{\frac{1-\gamma}{1-\frac{1}{\psi}}}\) with a small weight.

> **Degeneracies:** \(\psi\to 1\) approaches additive/separable (log-like); \(\gamma\to 1\) reduces risk curvature; recipe reduces toward CRRA smoothly.

---

## 5) ACTOR & CRITIC (Z-functions, distributions, exact log-probs)
### 5.1 Actor \(f_\theta\)
**Input:** \(s_t \in \mathbb{R}^{1+d+n}\) → shared backbone → three heads:
- Consumption pre-activation mean \(\mu_c := z_c \in \mathbb{R}\) and log-std \(\ell_c \in \mathbb{R}\); set \(\sigma_c := \mathrm{softplus}(\ell_c)+\sigma_{\min}\).  
  - Sample pre-squash \(y_c \sim \mathcal{N}(\mu_c, \sigma_c^2)\), then **squash**: \(c_t := \sigma(y_c) \in (0,1)\), where \(\sigma(\cdot)\) is the logistic sigmoid.
- Risky weights logits \(z_w \in \mathbb{R}^n\); **Dirichlet** concentration \(\alpha := \mathrm{softplus}(z_w) + \varepsilon_{\mathrm{dir}}\) → sample \(w_t \sim \mathrm{Dir}(\alpha)\).
- Deterministic eval: \(c_t := \sigma(\mu_c)\), \(w_t := \alpha / \sum_i \alpha_i\).

### 5.2 Exact log-probabilities (for PPO ratio)
- For consumption (squashed Gaussian): with \(y_c = \mathrm{logit}(c_t)\) and Jacobian \(\left|\frac{dy}{dc}\right| = \frac{1}{c_t (1-c_t)}\):
\[
\log p(c_t\mid s_t) \;=\; \log \mathcal{N}(y_c; \mu_c, \sigma_c^2) \;-\; \log\!\big( c_t(1-c_t) \big).
\]
- For weights (Dirichlet):
\[
\log p(w_t\mid s_t) \;=\; \log \Gamma\!\Big(\textstyle\sum_i \alpha_i\Big) \;-\; \sum_i \log \Gamma(\alpha_i) \;+\; \sum_i (\alpha_i-1)\log w_t[i].
\]
- Joint: \(\log \pi_\theta(a_t\mid s_t) = \log p(c_t\mid s_t) + \log p(w_t\mid s_t)\) (store at collection).

### 5.3 Critic \(g_\psi\) (**two heads for EZ**)
- Input: \(s_t\). Shared backbone → two scalar heads: \(\hat z_t\) and \(\hat y_t\).  
- Only \(\hat z_t\) is used in advantages/GAE; \(\hat y_t\) feeds the \(z\)-target via §4.2.

---

## 6) ENVIRONMENT STEP (FULL SEQUENCE)
At each time \(t\):
1. Build state \(s_t = \mathrm{concat}(\tilde W_t, \tilde x_t, w_{t-1})\) (after FD alignment, see §3.3).
2. Actor forward: get \(\mu_c, \sigma_c, \alpha\) → sample \(c_t, w_t\); compute and **store** \(\log \pi_{\theta_{\text{old}}}(a_t\mid s_t)\) exactly (§5.2).
3. Consumption: \(C_t := c_t \cdot W_t\).
4. Observe \(R[t+1], R_f[t+1]\); turnover \(\lVert w_t - w_{t-1}\rVert_1\); cost \(\mathrm{TC}_t := \kappa W_t \lVert w_t - w_{t-1}\rVert_1\).
5. Wealth update via §1.4 → \(W_{t+1}\); update max \(M_{t+1}\).
6. Build next features \(x_{t+1}\) (post-FracDiff), standardize \(\to \tilde x_{t+1}\).
7. Next state: \(s_{t+1} := \mathrm{concat}(W_{t+1}/M_{t+1}, \tilde x_{t+1}, w_t)\).

---

## 7) REWARDS (EXTERNAL EZ FLOW, INTRINSIC ICM)
### 7.1 External reward (EZ flow term in \(z\)-space)
\[
r_t^{\mathrm{ext}} := (1-\beta)\, C_t^{\,1-\frac{1}{\psi}}.
\]

### 7.2 Intrinsic curiosity reward (ICM)
Encoders and models (as in baseline):
- State encoder \(\phi(s) \in \mathbb{R}^m\); forward model \(f(\phi(s_t), \psi(a_t)) \to \hat \phi_{t+1}\); (optional) inverse model for stability.
- Action embedding \(\psi(a_t) := \mathrm{concat}\big( \mathrm{logit}(c_t), w_t \big)\).
- Intrinsic reward:
\[
r_t^{\mathrm{int}} := \eta \, \big\lVert\, \phi(s_{t+1}) - \hat \phi_{t+1} \,\big\rVert_2^2.
\]
- ICM losses: \(L_{\mathrm{fwd}} = \lVert\cdot\rVert_2^2\), \(L_{\mathrm{inv}} = -\big[ \log \mathcal{N}(y_c; \hat\mu_c, \hat\sigma_c^2) + \log \mathrm{Dir}(w_t; \hat\alpha) \big]\) (optional), \(L_{\mathrm{ICM}} = L_{\mathrm{fwd}} + \lambda_{\mathrm{inv}} L_{\mathrm{inv}}\).

### 7.3 Total reward
\(r_t := r_t^{\mathrm{ext}} + r_t^{\mathrm{int}}\) (what enters the advantage computation).

---

## 8) ADVANTAGES, TARGETS, AND LOSSES (EZ version)
### 8.1 EZ TD residual and GAE (in \(z\)-space)
- Build bootstrap target for \(z\) using next-state head:
\[
T^{(z)}_t \;=\; (1-\beta)\, C_t^{\,1-\frac{1}{\psi}} \;+\; \beta \,\Big(\hat y_{t+1}\Big)^{\frac{1-\frac{1}{\psi}}{\,1-\gamma\,}}.
\]
- Define TD residual:
\[
\delta_t^{\mathrm{EZ}} \;:=\; r_t \;+\; \beta\big(T^{(z)}_t - r_t^{\mathrm{ext}}\big) \;-\; \hat z_t.
\]
Intuition: \(T^{(z)}\) already contains the \(\beta\)-weighted continuation; \(r_t^{\mathrm{ext}}\) is the immediate \(z\)-flow.  
- Compute **GAE(\(\lambda\))** on \(\delta_t^{\mathrm{EZ}}\) exactly as baseline GAE (backward recursion).  
- Advantages are normalized per batch and used in PPO unchanged.

### 8.2 Losses
- **Policy (clipped PPO):** exactly baseline with advantages from §8.1.  
- **Value (z-head):** \( L_{\mathrm{value}} = \tfrac{1}{2}(\hat z_t - T^{(z)}_t)^2 \).  
- **Entropy:** unchanged (Gaussian + Dirichlet).  
- **ICM:** unchanged.  
- **Total:**  
\[
L_{\mathrm{total}} \;=\; L_{\mathrm{PPO}} \;+\; c_v L_{\mathrm{value}} \;+\; \beta_{\mathrm{ent}} L_{\mathrm{ent}} \;+\; c_{\mathrm{icm}} L_{\mathrm{ICM}}.
\]

---

## 9) TRAINING PROCEDURE (COLLECT → TARGETS → PPO)
### 9.1 Hyperparameters (additions/changes)
- **EZ:** choose \(\gamma \in \{5,10\}\), \(\psi \in \{0.5,1.0,1.5\}\), \(\beta \in [0.95,0.999]\).
- **FracDiff:** \(d_{\text{target}}\) init \(0.3\)–\(0.5\) within \([0,1]\), tolerance \(10^{-4}\), \(K_{\max} \in [1024,4096]\) (match horizon).
- **RL:** keep PPO \(\lambda\), clip, epochs, minibatch, lrs same initially.

### 9.2 Rollout collection (unchanged mechanics)
- Collect tuples \(\big(s_t, a_t=(c_t,w_t), r_t, s_{t+1}, \log \pi_{\theta_{\text{old}}}\big)\) where \(r_t\) includes EZ flow + curiosity.  
- Align time by dropping first \(K\) steps due to FracDiff.

### 9.3 Target building & PPO update
- For each step, compute \(T^{(z)}_t\), \(\delta_t^{\mathrm{EZ}}\), GAE, and \(z\)-value loss.  
- Recompute current \(\log \pi_\theta\) exactly (§5.2); perform clipped PPO with entropy and ICM losses.  
- After epochs, set \(\theta_{\text{old}} \leftarrow \theta\).

### 9.4 Evaluation (deterministic)
- Use \(c_t := \sigma(\mu_c)\), \(w_t := \alpha/\sum_i \alpha_i\).  
- Recover EZ value via inverse transform for reporting: \(\hat V_t = \hat z_t^{\,1/(1-\frac{1}{\psi})}\).  
- Report PnL, CAGR, MDD, Calmar, turnover, and \(\hat V_0\).

---

## 10) DIAGNOSTICS, CHECKS, & ABALATIONS
- **EZ sanity:** as \(\psi\to 1\) or \(\gamma\to 1\), curves and training behavior should smoothly approach separable/CRRA.  
- **Scale hygiene:** track \(\hat z_t, \hat y_t\) magnitudes; clamp/normalize if exploding.  
- **FracDiff:** monitor learned \(d_{\text{target}}\) trajectory; inspect ACF/PACF of FD outputs; avoid non-stationary drift.  
- **Alignment:** verify all post-FD tensors drop the first \(K\) steps; shapes of policy/value/ICM batches match.  
- **Ablations:** (i) turn off FracDiff (identity) to test EZ alone; (ii) \(\psi\) grid with fixed \(\gamma\); (iii) compare CRRA vs EZ at matched \(\gamma\) with \(\psi\approx 1\).

---

## 11) MINIMAL MIGRATION CHECKLIST
- [ ] Expose \(\gamma, \psi, \beta\) in config; leave PPO hypers unchanged initially.  
- [ ] Critic: switch to **two heads** \((\hat z, \hat y)\); keep shared backbone.  
- [ ] Reward pipe: compute \(r_t^{\mathrm{ext}}=(1-\beta)C^{\,1-1/\psi}\); add curiosity as before → \(r_t\).  
- [ ] Targets: build \(T^{(z)}\) with next-state \(\hat y\); compute \(\delta^{\mathrm{EZ}}\) and GAE in \(z\)-space.  
- [ ] Insert **Learnable FracDiff** before feature builder; shift by \(K\).  
- [ ] Log \(d_{\text{target}}, K\), ACF diagnostics, and \(\hat z,\hat y\) summaries.

---

## 12) GLOSSARY
- \(C_t\): dollar consumption; \(W_t\): wealth; \(c_t\): consumption rate.  
- \(\gamma\): risk aversion; \(\psi\): EIS; \(\beta\): discount.  
- \(V_t\): EZ lifetime utility; \(z(V)=V^{\,1-1/\psi}\); \(y(V)=V^{\,1-\gamma}\).  
- \(d_{\text{target}}\): fracdiff memory parameter; \(K\): kernel truncation length.  
- ICM: Intrinsic Curiosity Module; \(\phi\): encoder; \(f\): forward model.

---

## 13) TL;DR (one-screen summary)
- **Objective change:** CRRA → **Epstein–Zin** with a two-head critic \((\hat z, \hat y)\), shaped external reward \((1-\beta)C^{\,1-1/\psi}\), and a one-step \(z\)-target using \(\hat y_{t+1}\).  
- **Feature change:** Insert **Learnable FracDiff** over returns; align time by kernel length \(K\).  
- **PPO/ICM:** identical machinery; only the value target and reward semantics change.  
- **Eval:** deterministic actions; report \(\hat V_0\) via inverse transform and standard trading metrics.