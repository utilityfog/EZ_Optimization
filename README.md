# SPEC: RL INVESTOR (PPO + ICM) with **Epstein–Zin** Preferences + **Learnable Fractional Differencing**
*A self-contained, implementation-ready specification. No code included. All math, shapes, distributions, rewards, targets, and losses are explicit.*

---

## 0) PURPOSE & SCOPE
This document **replaces CRRA** with **Epstein–Zin (EZ)** recursive preferences and **adds a Learnable Fractional Differencing (FracDiff) layer** in the feature pipeline, while preserving the **PPO + ICM** training stack and environment mechanics. It is **drop-in**: policy parameterization, buffers, rollout loop, exact log-probabilities, and PPO machinery remain intact. Only the **preference aggregator** (reward/value semantics) and **feature memory module** (FracDiff) change.

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
- Discrete time $\(t = 0,1,2,\ldots,T-1\)$ (episode length $T$).  
- Number of risky assets $\(n \ge 1\)$.
- Market data (known up to $\(t\)$ when acting):
  - Risky **gross** returns: $\(R[t] \in \mathbb{R}^n\)$ (e.g., $1.01 = +1%$).
  - Risk-free **gross** return: $\(R_f[t] \in \mathbb{R}\)$.

### 1.2 Wealth, consumption, normalization
- Wealth at start of step $\(t\)$: $\(W_t > 0\)$.
- Consumption fraction (action): $\(c_t \in (0,1)\)$; **dollar consumption** $\(C_t := c_t \cdot W_t\)$.
- Running max wealth $\(M_t := \max_{0\le \tau \le t} W_\tau\)$; normalized wealth $\(\tilde W_t := W_t / M_t \in (0,1]\)$.

### 1.3 Portfolio, turnover, transaction cost
- Risky-asset weights (action): $\(w_t \in \Delta^n\)$ (simplex, nonnegative, $\(\sum_{i=1}^n w_t[i] = 1 \)$ ).  
- Implicit cash weight: $\(w_{\text{cash},t} := 1 - \sum_{i=1}^n w_t[i]\)$ (nonnegative by construction).
- Turnover: $\(\mathrm{turnover_t} := \lVert w_t - w_{t-1} \rVert_1\)$.
- Transaction-cost coefficient: $\(\kappa \ge 0\)$.
- Dollar cost: $\(\mathrm{TC}_t := \kappa \cdot W_t \cdot \mathrm{turnover}_t\)$ (paid immediately at $\(t\)$ ).

### 1.4 Budget identity (wealth transition)
Let risky **excess** return $\(\tilde R[t+1] := R[t+1] - R_f[t+1]\cdot \mathbf{1}\)$.
- Gross growth factor:
$$\(
\[
G_{t+1} := (1 - c_t)*\big( R_f[t+1] + w_t^{\top} \tilde R[t+1] \big) - \kappa \lVert w_t - w_{t-1}\rVert_1.
\]
\)$$
- Next wealth: $\(W_{t+1} := W_t \cdot G_{t+1}\)$. Safety floors may clip $\(G_{t+1}\)$ to $\(\varepsilon_g>0\)$.

---

## 2) OBSERVATIONS, FEATURES, STATE (BASELINE PIPELINE)
### 2.1 Observables at time \(t\)
- $\(W_t\)$, $\(w_{t-1}\)$, and a causal feature vector $\(x_t \in \mathbb{R}^d\)$ built **only** from data $\(\le t\)$.
- Standardize $\(x_t\)$ via train-set $\((\mu,\sigma)\)$ to $\(\tilde x_t\)$ (store $\(\mu,\sigma\)$ from training only).

### 2.2 State to networks
- **State:** $\(s_t := \mathrm{concat}\big( \tilde W_t, \tilde x_t, w_{t-1} \big) \in \mathbb{R}^{1+d+n}\)$ (fixed order).

---

## 3) **Learnable Fractional Differencing** (returns-domain feature module)
### 3.1 Goal & parameter
Learn a memory depth $\(d_{\text{target}} \in [d_{\min}, d_{\max}]\)$ (e.g., $\([0,1]\)$ ) that controls the fractional differencing of returns to **capture long memory** while promoting **stationarity**.

### 3.2 Placement in pipeline
- Input raw **log-returns** per asset: $\(r_t \in \mathbb{R}^n\)$ (or windows).  
- Apply a FracDiff operator with effective exponent $\(d_{\text{eff}}\)$:
  - **Mode “direct”**: apply $\((1 - L)^{d_{\text{target}}}\)$ to returns.  
  - **Mode “price\_equiv”**: apply $\((1 - L)^{d_{\text{target}} - 1}\)$ to returns (equivalent to price fracdiff of $\(d_{\text{target}}\)$ without reconstructing prices).
- Truncate the kernel to length $\(K\)$ (auto-chosen from $\(d_{\text{eff}}\)$ and a tolerance). Outputs lose the first $\(K\)$ steps.

### 3.3 State augmentation & alignment
- Build usual statistics **from** the FD output (lags, MAs, vol, PCA, cross-sectional transforms).  
- **Shift** all time-aligned targets by $\(K\)$ (drop first $\(K\)$ steps) so shapes match.  
- Optionally append $\(\mathrm{stop\_grad}(d_{\text{target}})\)$ and $\(K\)$ as scalar features so the policy/critic can adapt to memory depth.

### 3.4 Regularization & constraints
- Keep $\(d_{\text{target}}\)$ within bounds via a sigmoid reparameterization.  
- Add a small L2 penalty if $\(d_{\text{target}}\)$ sticks to the bounds.  
- Optional “whiteness” regularizer: penalize low-lag autocorrelation of FD residuals to avoid over-memory.

> **Everything backpropagates end-to-end** because kernel weights are differentiable functions of $\(d_{\text{eff}}\)$.

---

## 4) **Epstein–Zin** Preferences (replace CRRA)
Let $\(\beta\in(0,1)\)$ be the subjective discount, $\(\gamma>0\)$ risk aversion, $\(\psi>0\)$ intertemporal elasticity (EIS). Define transforms:
- $\( z(V) := V^{\,1-\frac{1}{\psi}} \)$   (EIS/consumption space)  
- $\( y(V) := V^{\,1-\gamma} \)$           (risk space)

### 4.1 EZ aggregator (Kreps–Porteus form)
For lifetime utility $\(V_t\)$ and consumption $\(C_t\)$:

$$\(
\[
V_t = \Big[(1-\beta)\, C_t^{\,1-\frac{1}{\psi}} + \beta \,\big( \mathbb{E}_t [ V_{t+1}^{\,1-\gamma} ] \big)^{\frac{1-\frac{1}{\psi}}{\,1-\gamma\,}} \Big]^{\frac{1}{\,1-\frac{1}{\psi}\,}}.
\]
\)$$

### 4.2 Practical RL parameterization (stable targets)
We **train in $\(z\)$-space** with a two-head critic predicting $\(\hat z_t \approx z(V_t)\)$ and $\(\hat y_t \approx y(V_t)\)$.  
- **External (shaped) reward:** $\( r_t^{\mathrm{ext}} := (1-\beta)\, C_t^{\,1-\frac{1}{\psi}} \)$.
- **One-step bootstrap target for $\(z\)$:**

$$\(
\[
T^{(z)}_t := (1-\beta)\, C_t^{\,1-\frac{1}{\psi}} + \beta \,\Big(\hat y_{t+1}\Big)^{\frac{1-\frac{1}{\psi}}{\,1-\gamma\,}}.
\]
\)$$

- **Value loss:** $\( L_{\mathrm{value}} := \tfrac{1}{2}\,\big(\hat z_t - T^{(z)}_t\big)^2 \)$.
- Optional **consistency** regularizer: encourage $\( \hat y_t \approx (\hat z_t)^{\frac{1-\gamma}{1-\frac{1}{\psi}}}\)$ with a small weight.

> **Degeneracies:** $\(\psi\to 1\)$ approaches additive/separable (log-like); $\(\gamma\to 1\)$ reduces risk curvature; recipe reduces toward CRRA smoothly.

---

## 5) ACTOR & CRITIC (Z-functions, distributions, exact log-probs)

**Dimensions and symbols used throughout this section**
- Number of risky assets: $\(n \ge 1\)$.  Feature dimension: $\(d \ge 1\)$.
- State at time $\(t\)$: $\(s_t \in \mathbb{R}^{1+d+n}\)$ is the concatenation
  $\(s_t := \mathrm{concat}\big(\tilde W_t,\ \tilde x_t,\ w_{t-1}\big)\)$,
  where $\(\tilde W_t = W_t/M_t\)$, $\(\tilde x_t\)$ is the standardized feature vector, and $\(w_{t-1}\in\mathbb{R}^n\)$ is the previous risky-weights vector on the simplex $\(\Delta^n\)$.
- Consumption fraction (action component): $\(c_t \in (0,1)\)$. Dollar consumption: $\(C_t := c_t W_t\)$.
- Risky weights (action component): $\(w_t \in \Delta^n = \{u\in\mathbb{R}_{\ge 0}^n:\sum_i u_i=1\}\)$.
- Hyperparameters for heads: $\(\sigma_{\min}>0\)$ (std floor), $\(\varepsilon_{\mathrm{dir}}>0\)$ (Dirichlet floor).

### 5.1 Actor $\(f_\theta\)$
Shared backbone on $\(s_t\)$ (MLP with chosen sizes/activations), followed by three heads:

- **Consumption head (squashed Gaussian).**
  - Pre-squash Normal parameters: mean $\(\mu_c \in \mathbb{R}\)$, log-std $\(\ell_c \in \mathbb{R}\)$, std $\(\sigma_c := \mathrm{softplus}(\ell_c)+\sigma_{\min}\)$.
  - Sample pre-squash $\(y_c \sim \mathcal{N}(\mu_c,\sigma_c^2)\)$, then **squash** $\(c_t := \sigma(y_c)=1/(1+e^{-y_c})\in(0,1)\)$.
  - Deterministic eval: $\(c_t := \sigma(\mu_c)\)$.

- **Risky-weights head (Dirichlet).**
  - Logits $\(z_w \in \mathbb{R}^n\)$; concentrations $\(\alpha := \mathrm{softplus}(z_w)+\varepsilon_{\mathrm{dir}}\in\mathbb{R}_{>0}^n\)$.
  - Sample $\(w_t \sim \mathrm{Dir}(\alpha)\)$; deterministic eval $\(w_t := \alpha/\sum_i \alpha_i\)$.

**Action log-probabilities (exact).**

- Let $\(y_c := \mathrm{logit}(c_t) = \log\big(\tfrac{c_t}{1-c_t}\big)\)$. Then the change-of-variables gives

$$\(
\[
\log p(c_t\mid s_t) = \log \mathcal{N}(y_c,\mu_c,\sigma_c^2) - \log \big(c_t(1-c_t)\big).
\]
\)$$

- For $\(w_t\sim\mathrm{Dir}(\alpha)\)$:

$$\(
\[
\log p(w_t\mid s_t) = \log \Gamma \Big(\textstyle\sum_{i=1}^n \alpha_i\Big) - \sum_{i=1}^n \log \Gamma(\alpha_i) + \sum_{i=1}^n (\alpha_i-1)\log w_t[i].
\]
\)$$

- **Joint** log-prob used by PPO:

$\(\log \pi_\theta(a_t\mid s_t) := \log p(c_t\mid s_t) + \log p(w_t\mid s_t)\)$.

### 5.3 Critic $\(g_\psi\)$ (two heads for EZ)
The critic takes $\(s_t\)$ and outputs two scalars:
- $\(\hat z_t \approx z(V_t)\)$ with $\(z(V):=V^{\,1-\frac{1}{\psi}}\)$.
- $\(\hat y_t \approx y(V_t)\)$ with $\(y(V):=V^{\,1-\gamma}\)$.
These are used to build the EZ bootstrap target and TD residual below.

---

## 6) ENVIRONMENT STEP (FULL SEQUENCE)

**Market data and costs**
- Risk-free gross return: $\(R_f[t] \in \mathbb{R_{>0}}\)$. Risky gross returns: $\(R[t] \in \mathbb{R_{>0}^n}\)$.
- Excess return: $\(\tilde R[t] := R[t] - R_f[t]\mathbf{1}\)$.
- Turnover and cost coefficient: turnover $\(\|w_t-w_{t-1}\|_1\)$, cost $\(\kappa\ge 0\)$.

**Wealth transition**
- Gross growth factor:

$$\(
\[
G_{t+1} := (1 - c_t)\big( R_f[t+1] + w_t^{\top} \tilde R[t+1] \big) - \kappa \lVert w_t - w_{t-1}\rVert_1.
\]
\)$$

- Next wealth: $\(W_{t+1} := W_t \cdot G_{t+1}\)$. Safety floor may clip $\(G_{t+1}\ge\varepsilon_g>0\)$.

**State update**
- Update running max $\(M_{t+1} := \max(M_t,W_{t+1})\)$.
- Build next features $\(\tilde x_{t+1}\)$ (post-FracDiff pipeline) and set
  $\(s_{t+1} := \mathrm{concat}(W_{t+1}/M_{t+1},\ \tilde x_{t+1},\ w_t)\)$.

---

## 7) REWARDS (EXTERNAL EZ FLOW, INTRINSIC ICM)

**EZ parameters:** discount $\(\beta\in(0,1)\)$, risk aversion $\(\gamma>0\)$, EIS $\(\psi>0\)$.

### 7.1 External reward (EZ flow term in $\(z\)$-space)

$$\(
\[
r_t^{\mathrm{ext}} := (1-\beta)\, C_t^{\,1-\frac{1}{\psi}}.
\]
\)$$

### 7.2 Intrinsic curiosity reward (ICM) — every variable defined

**Networks and dimensions**
- Choose embedding dims $\(m\)$ ( e.g., $\(64\)$ ), hidden widths $\(E,F\)$ ( e.g., $\(128\)$ ).
- **State encoder** $\(\phi_\omega:\mathbb{R}^{1+d+n}\to\mathbb{R}^m\)$:
  - $\(e1 := \mathrm{GELU}(W_{e1}s_t+b_{e1})\)$, $\(W_{e1}\in\mathbb{R}^{E\times(1+d+n)}\)$.
  - $\(e2 := \mathrm{GELU}(W_{e2}e1+b_{e2})\)$, $\(W_{e2}\in\mathbb{R}^{E\times E}\)$.
  - $\(\phi(s_t) := W_{eo}e2 + b_{eo} \in \mathbb{R}^m\)$.
- **Action embedding** for continuous $\(a_t=(c_t,w_t)\)$:
  - $\(y_c := \mathrm{logit}(c_t)=\log(\tfrac{c_t}{1-c_t})\)$.
  - $\(\psi(a_t) := \mathrm{concat}(y_c,\ w_t) \in \mathbb{R}^{1+n}\)$.
- **Forward model** $\(f_\omega:\mathbb{R}^m\times\mathbb{R}^{1+n}\to\mathbb{R}^m\)$:
  - $\(u1 := \mathrm{GELU}\big(W_{f1}\,\mathrm{concat}(\phi(s_t),\psi(a_t))+b_{f1}\big)\)$, $\(W_{f1}\in\mathbb{R}^{F\times(m+1+n)}\)$.
  - $\(u2 := \mathrm{GELU}(W_{f2}u1+b_{f2})\)$, $\(W_{f2}\in\mathbb{R}^{F\times F}\)$.
  - $\(\hat\phi_{t+1} := f(\phi(s_t),a_t) := W_{fo}u2+b_{fo} \in \mathbb{R}^m\)$.
- **(Optional) Inverse model** $\(g_\omega:\mathbb{R}^m\times\mathbb{R}^m\to\)$ action params:
  - Given $\(\phi(s_t)\), \(\phi(s_{t+1})\)$, predict $\(\hat\mu_c,\hat\sigma_c\)$ for $\(y_c\)$ and $\(\hat\alpha\in\mathbb{R}_{>0}^n\)$ for Dirichlet over $\(w_t\)$.

**Intrinsic reward and ICM losses**
- Embeddings: $\(\phi_t:=\phi(s_t)\)$, $\(\phi_{t+1}:=\phi(s_{t+1})\)$, prediction $\(\hat\phi_{t+1}\)$.
- Reward scale $\(\eta>0\)$ ( e.g., $\(10^{-3}\)$ ).
- Intrinsic reward:

$$\(
\[
r_t^{\mathrm{int}} := \eta \, \big\lVert\, \phi_{t+1} - \hat \phi_{t+1} \,\big\rVert_2^2.
\]
\)$$

- Forward loss: $\(L_{\mathrm{fwd}}(\omega) := \big\lVert \phi_{t+1} - \hat\phi_{t+1} \big\rVert_2^2\)$.
- Inverse loss (optional):
  - $\(L_{\mathrm{inv}}(\omega) := -\big[\log \mathcal{N}(y_c\hat\mu_c,\hat\sigma_c^2)\ +\ \log \mathrm{Dir}(w_t\hat\alpha)\big]\)$.
- Combine with weight $\(\lambda_{\mathrm{inv}}\ge 0\)$:
  - $\(L_{\mathrm{ICM}} := L_{\mathrm{fwd}} + \lambda_{\mathrm{inv}} L_{\mathrm{inv}}\)$.

### 7.3 Total reward used by PPO
$\(r_t := r_t^{\mathrm{ext}} + r_t^{\mathrm{int}}\)$.

---

## 8) ADVANTAGES, TARGETS, AND LOSSES (EZ version)

**EZ one-step target and TD in $\(z\)$-space**
- One-step bootstrap target ( uses next-state $\(\hat y_{t+1}\)$ ):

$$\(
\[
T^{(z)}_t = (1-\beta)\, C_t^{\,1-\frac{1}{\psi}} + \beta \,\Big(\hat y_{t+1}\Big)^{\frac{1-\frac{1}{\psi}}{\,1-\gamma\,}}.
\]
\)$$

- EZ TD residual:

$$\(
\[
\delta_t^{\mathrm{EZ}} := r_t + \beta\big(T^{(z)}_t - r_t^{\mathrm{ext}}\big) - \hat z_t.
\]
\)$$

**GAE advantages**
- GAE parameter $\(\lambda\in[0,1]\)$. Compute advantages $\(\tilde A_t\)$ by the standard backward recursion on $\(\delta_t^{\mathrm{EZ}}\)$, then **normalize** over the batch.

**PPO policy loss (clipped)**
- Store behavior log-prob $\(\log \pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)\)$ during rollout.
- Recompute $\(\log \pi_\theta(a_t\mid s_t)\)$ at update time via §5.1/5.2.
- Importance ratio $\(r_t(\theta):=\exp\big(\log \pi_\theta - \log \pi_{\theta_{\mathrm{old}}}\big)\)$.
- Clip parameter $\(\varepsilon\in(0,1)\)$ ( e.g., $\(0.2\)$ ).
- **Definition of $\(L_{\mathrm{PPO}}\)$ (to minimize):**

$$\(
\[
L_{\mathrm{PPO}} := -\,\mathbb{E}_t\Big[\min \big(r_t(\theta)\,\tilde A_t,\ \mathrm{clip}(r_t(\theta),1-\varepsilon,1+\varepsilon)\,\tilde A_t\big)\Big].
\]
\)$$

**Value loss (z-head)**
- $\(L_{\mathrm{value}} := \tfrac{1}{2}\big(\hat z_t - T^{(z)}_t\big)^2\)$.

**Entropy loss**
- Digamma $\(\psi_0(\cdot)\)$ is the derivative of $\(\log\Gamma(\cdot)\)$.
- Entropy of pre-squash Normal $\(y_c \sim \mathcal{N}(\mu_c,\sigma_c^2)\)$:

$$\(
\[
H_c := \tfrac{1}{2}\,\log\big(2\pi e\,\sigma_c^2\big).
\]
\)$$

- Entropy of Dirichlet with $\(\alpha\in\mathbb{R}_{>0}^n\)$:

$$\(
\[
H_w := \log \Gamma \Big(\textstyle\sum_i \alpha_i\Big) - \sum_i \log \Gamma(\alpha_i) + \Big(\textstyle\sum_i (\alpha_i-1)\Big)\psi_0\Big(\textstyle\sum_i \alpha_i\Big) - \sum_i (\alpha_i-1)\psi_0(\alpha_i).
\]
\)$$

- **Entropy loss to minimize**:

$$\(
\[
L_{\mathrm{ent}} := -(H_c + H_w).
\]
\)$$

**ICM loss**
- Already defined above: $\(L_{\mathrm{ICM}} := L_{\mathrm{fwd}} + \lambda_{\mathrm{inv}} L_{\mathrm{inv}}\)$.

**Final objective and all coefficients (all defined here)**
- Coefficient $\(c_v>0\)$ weights the value loss.
- Coefficient $\(\beta_{\mathrm{ent}}>0\)$ weights the entropy loss.
- Coefficient $\(c_{\mathrm{icm}}>0\)$ weights the ICM loss.

$$\(
\[
L_{\mathrm{total}} = L_{\mathrm{PPO}} + c_v L_{\mathrm{value}} + \beta_{\mathrm{ent}} L_{\mathrm{ent}} + c_{\mathrm{icm}} L_{\mathrm{ICM}}.
\]
\)$$

**How to compute the coefficients in practice (turnkey rules)**

- **Initialization (works out of the box):**
  - $\(c_v = 1.0\)$, $\(\beta_{\mathrm{ent}} = 10^{-3}\)$, $\(c_{\mathrm{icm}} = 0.5\)$, $\(\lambda_{\mathrm{inv}}=0\)$ (enable inverse later if needed).
- **Adaptive entropy weight** (keep exploration near a target):
  - Choose a target total entropy $\(H_{\text{target}}\)$ ( sum of $\(H_c\)$ and $\(H_w\)$ ). Each epoch, update

    $$\(
    \[
    \beta_{\mathrm{ent}} \leftarrow \mathrm{clip}\Big(\beta_{\mathrm{ent}}\cdot \exp\big(\tau\,[\,H_{\text{target}}-(H_c{+}H_w)\,]\big),\ \beta_{\min},\ \beta_{\max}\Big),
    \]
    \)$$

    with small gain $\(\tau\in[10^{-3},10^{-2}]\)$, and bounds $\(\beta_{\min}=10^{-5}\)$, $\(\beta_{\max}=10^{-2}\)$.
- **Value-loss weight via scale matching** (keep terms numerically comparable):
  - Maintain running RMS for the unclipped policy term $\(U_t:=r_t(\theta)\tilde A_t\)$ and the squared value error $\(V_t:=\tfrac{1}{2}(\hat z_t-T^{(z)}_t)^2\)$ (stop-grad on denominators). Set

    $$\(
    \[
    c_v \leftarrow \mathrm{clip}\Bigg(\frac{\mathrm{RMS}[U_t]}{\mathrm{RMS}[V_t]+\epsilon},\ c_{v,\min},\ c_{v,\max}\Bigg),
    \]
    \)$$

    with $\(c_{v,\min}=0.25\)$, $\(c_{v,\max}=2.0\)$, $\(\epsilon=10^{-8}\)$.
- **ICM weight by intrinsic-share target** (keep curiosity proportion controlled):
  - Pick $\(p_{\text{int}}\in[0,0.3]\)$. After each epoch compute
    $\(q:=\frac{\sum_t r_t^{\mathrm{int}}}{\sum_t (r_t^{\mathrm{ext}}+r_t^{\mathrm{int}})}\)$.
    Update $\(c_{\mathrm{icm}}\)$ multiplicatively:

    $$\(
    \[
    c_{\mathrm{icm}} \leftarrow \mathrm{clip}\Big(c_{\mathrm{icm}}\cdot \exp(\rho\,[\,p_{\text{int}}-q\,]),\ c_{\mathrm{icm},\min},\ c_{\mathrm{icm},\max}\Big),
    \]
    \)$$

    with small gain $\(\rho\in[10^{-3},10^{-2}]\)$ and bounds $\(c_{\mathrm{icm},\min}=0.05\)$, $\(c_{\mathrm{icm},\max}=2.0\)$.

---

## 9) TRAINING PROCEDURE (COLLECT → TARGETS → PPO)
### 9.1 Hyperparameters (additions/changes)
- **EZ:** choose $\(\gamma \in \{5,10\}\)$, $\(\psi \in \{0.5,1.0,1.5\}\)$, $\(\beta \in [0.95,0.999]\)$.
- **FracDiff:** $\(d_{\text{target}}\)$ init $\(0.3\)$ – $\(0.5\)$ within $\([0,1]\)$, tolerance $\(10^{-4}\)$, $\(K_{\max} \in [1024,4096]\)$ (match horizon).
- **RL:** keep PPO $\(\lambda\)$, clip, epochs, minibatch, lrs same initially.

### 9.2 Rollout collection (unchanged mechanics)
- Collect tuples $\(\big(s_t, a_t=(c_t,w_t), r_t, s_{t+1}, \log \pi_{\theta_{\text{old}}}\big)\)$ where $\(r_t\)$ includes EZ flow + curiosity.  
- Align time by dropping first $\(K\)$ steps due to FracDiff.

### 9.3 Target building & PPO update
- For each step, compute $\(T^{(z)}_t\)$, $\(\delta_t^{\mathrm{EZ}}\)$, GAE, and $\(z\)$-value loss.  
- Recompute current $\(\log \pi_\theta\)$ exactly (§5.2); perform clipped PPO with entropy and ICM losses.  
- After epochs, set $\(\theta_{\text{old}} \leftarrow \theta\)$.

### 9.4 Evaluation (deterministic)
- Use $\(c_t := \sigma(\mu_c)\)$, $\(w_t := \alpha/\sum_i \alpha_i\)$.  
- Recover EZ value via inverse transform for reporting: $\(\hat V_t = \hat z_t^{\,1/(1-\frac{1}{\psi})}\)$.  
- Report PnL, CAGR, MDD, Calmar, turnover, and $\(\hat V_0\)$.

---

## 10) DIAGNOSTICS, CHECKS, & ABALATIONS
- **EZ sanity:** as $\(\psi\to 1\)$ or $\(\gamma\to 1\)$, curves and training behavior should smoothly approach separable/CRRA.  
- **Scale hygiene:** track $\(\hat z_t, \hat y_t\)$ magnitudes; clamp/normalize if exploding.  
- **FracDiff:** monitor learned $\(d_{\text{target}}\)$ trajectory; inspect ACF/PACF of FD outputs; avoid non-stationary drift.  
- **Alignment:** verify all post-FD tensors drop the first $\(K\)$ steps; shapes of policy/value/ICM batches match.  
- **Ablations:** (i) turn off FracDiff (identity) to test EZ alone; (ii) $\(\psi\)$ grid with fixed $\(\gamma\)$; (iii) compare CRRA vs EZ at matched $\(\gamma\)$ with $\(\psi\approx 1\)$.

---

## 11) MINIMAL MIGRATION CHECKLIST
- [ ] Expose $\(\gamma, \psi, \beta\)$ in config; leave PPO hypers unchanged initially.  
- [ ] Critic: switch to **two heads** $\((\hat z, \hat y)\)$; keep shared backbone.  
- [ ] Reward pipe: compute $\(r_t^{\mathrm{ext}}=(1-\beta)C^{\,1-1/\psi}\)$; add curiosity as before → $\(r_t\)$.  
- [ ] Targets: build $\(T^{(z)}\)$ with next-state $\(\hat y\)$; compute $\(\delta^{\mathrm{EZ}}\)$ and GAE in $\(z\)$-space.  
- [ ] Insert **Learnable FracDiff** before feature builder; shift by $\(K\)$.  
- [ ] Log $\(d_{\text{target}}, K\)$, ACF diagnostics, and $\(\hat z,\hat y\)$ summaries.

---

## 12) GLOSSARY
- $\(C_t\)$: dollar consumption; $\(W_t\)$: wealth; $\(c_t\)$: consumption rate.  
- $\(\gamma\)$: risk aversion; $\(\psi\)$: EIS; $\(\beta\)$: discount.  
- $\(V_t\)$: EZ lifetime utility; $\(z(V)=V^{\,1-1/\psi}\)$; $\(y(V)=V^{\,1-\gamma}\)$.  
- $\(d_{\text{target}}\)$: fracdiff memory parameter; $\(K\)$: kernel truncation length.  
- ICM: Intrinsic Curiosity Module; $\(\phi\)$: encoder; $\(f\)$: forward model.

---

## 13) TL;DR (one-screen summary)
- **Objective change:** CRRA → **Epstein–Zin** with a two-head critic $\((\hat z, \hat y)\)$, shaped external reward $\((1-\beta)C^{\,1-1/\psi}\)$, and a one-step $\(z\)$-target using $\(\hat y_{t+1}\)$.  
- **Feature change:** Insert **Learnable FracDiff** over returns; align time by kernel length $\(K\)$.  
- **PPO/ICM:** identical machinery; only the value target and reward semantics change.  
- **Eval:** deterministic actions; report $\(\hat V_0\)$ via inverse transform and standard trading metrics.