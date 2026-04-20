# Debiased Machine Learning in High-Dimensional Discrete Settings

This project studies **Double / Debiased Machine Learning (DML)** in high-dimensional settings with **discrete and grouped covariates**, focusing on how nuisance function estimators behave under varying sparsity, dimensionality, and confounding structures.

We implement and evaluate the **Augmented Inverse Probability Weighting (AIPW)** estimator with **cross-fitted machine learning nuisance models**, and systematically compare a wide range of Machine Learners in simulated causal inference environments.

## Configuration
- Run in Terminal the command "pip install -e ."
- Generalizing group structure: frontends, backends, (middleend for code serving as a conceptual bridge)
- To run the core pipeline: enter "src/AIPW"

---

# Research Objective

We investigate how nuisance estimation impacts causal inference in high-dimensional discrete settings:

- Moderate to High-dimensional regimes
- Group-structured covariates (one-hot encoded)
- Latent confounding in treatment assignment
- Sparse vs dense signal regimes

# Methodology

## 1. AIPW Estimator

The main estimator is the **Augmented Inverse Probability Weighting (AIPW)** estimator:

$$\psi_i = \frac{D_i (Y_i - m_1(X_i))}{e(X_i)} - \frac{(1 - D_i)(Y_i - m_0(X_i))}{1 - e(X_i)}+ m_1(X_i) - m_0(X_i)$$

The treatment effect estimator is:

\[
\hat{\tau} = \frac{1}{n} \sum_{i=1}^{n} \psi_i
\]

Variance estimate:

\[
\widehat{\mathrm{Var}}(\hat{\tau}) = \frac{1}{n} \mathrm{Var}(\psi_i)
\]

Confidence interval:

\[
\hat{\tau} \pm 1.96 \cdot \hat{\mathrm{SE}}
\]

---

## 2. Cross-Fitting Procedure

To ensure orthogonality and avoid overfitting bias:

- K-fold sample splitting
- Separate training of nuisance models:
  - Outcome models: \( m_0(X), m_1(X) \)
  - Propensity model: \( e(X) \)

---

## 3. Monte Carlo Simulation Framework

We evaluate estimator performance via parallel Monte Carlo:

Metrics computed:

- Bias:
\[
\mathbb{E}[\hat{\tau}] - \tau
\]

- Variance:
\[
\frac{1}{n_{\text{sim}}-1} \sum_{k=1}^{n_{\text{sim}}} \left(\hat{\tau}_k - \frac{1}{n_{\text{sim}}} \sum_{j=1}^{n_{\text{sim}}} \hat{\tau}_j\right)^2
\]

- Avg. CI width:
\[
\frac{1}{n_{\text{sim}}} \sum_{k=1}^{n_{\text{sim}}} \left( \hat{\tau}_k^{\text{upper}} - \hat{\tau}_k^{\text{lower}} \right)
\]

- Coverage:
\[
\mathbb{P}(\tau \in CI_{95\%})
\]

- RMSE:
\[
\sqrt{\mathbb{E}[(\hat{\tau} - \tau)^2]}
\]

- Mean:
\[
\frac{1}{n_{\text{sim}}} \sum_{k=1}^{n_{\text{sim}}} \hat{\tau}_k
\]

---

## 4. Machine Learning Models

### Linear Models
- OLS
- Ridge Regression
- Lasso
- Elastic Net
- Fused Lasso (CVXPY structured sparsity)

### Tree Models
- Random Forest
- Extra Trees
- Honest Forest

## Boosted Decision-Tree Models
- Gradient Boosting
- XGBoost
- CatBoost

### Kernel Methods
- Kernel Ridge Regression
- Support Vector Regression

### Neural Networks
- Multi-Layer Perceptron with scaling pipeline

---

## 5. High-Dimensional Discrete DGP

Covariates are group-structured:

\[
X \in \{0,1\}^{n \times G}
\]

Each observation belongs to exactly one group.

### Outcome model:

\[
Y_i = D_i Y_i(1) + (1 - D_i) Y_i(0)
\]

\[
Y_i(0) = \mu_{g(i)} + \varepsilon_i
\quad , \quad
Y_i(1) = \mu_{g(i)} + \tau + \varepsilon_i
\]

### Treatment assignment:

\[
D_i \sim \mathrm{Bernoulli}(p_{g(i)})
\]

where:

- \( \mu_g \): group-level baseline effects  
- \( p_g \): group-level propensity (confounded via latent factor)

---

## 6. Latent Confounding Structure
Group-level parameters are generated as:

$$
\beta_g = \alpha s_g + \epsilon_{\beta}, \quad \epsilon_{\beta} \sim \mathcal{N}(0, \sigma_{\beta}^2)
$$

$$
\rho_g = \gamma s_g + \epsilon_{\rho}, \quad \epsilon_{\rho} \sim \mathcal{N}(0, \sigma_{\rho}^2)
$$

The propensity score is defined via a bounded transformation:

$$
p_g = \sigma(\rho_g)
$$

where \( \sigma(\cdot) \) denotes the logistic function.

This induces **strong correlation between treatment and outcome mechanisms**.