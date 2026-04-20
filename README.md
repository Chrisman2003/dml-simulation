# Debiased Machine Learning in High-Dimensional Discrete Settings

This project studies **Double / Debiased Machine Learning (DML)** in high-dimensional settings with **discrete and grouped covariates**, focusing on how nuisance function estimators behave under varying sparsity, dimensionality, sample size, and confounding structures.

We implement and evaluate the **Augmented Inverse Probability Weighting (AIPW)** estimator with **cross-fitted machine learning nuisance models**, and systematically compare its performance under these machine learners with synthetic data-generating processes.

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

$$
\hat{\theta}_{\text{AIPW}} = \frac{1}{n} \sum_{i=1}^{n}\left[\frac{D_i (Y_i - m_1(X_i))}{e(X_i)} - \frac{(1 - D_i)(Y_i - m_0(X_i))}{1 - e(X_i)}+ m_1(X_i) - m_0(X_i)\right]
$$


With Confidence interval:

$$
CI = \hat{\theta}_{\text{AIPW}} \pm 1.96 \cdot \widehat{\mathrm{SE}}\big(\hat{\theta}_{\text{AIPW}}\big)
$$
---

## 2. Cross-Fitting Procedure

To ensure orthogonality and avoid overfitting bias:

- K-fold sample splitting
- Separate training of nuisance models:
  - Outcome models: \( m_0(X), m_1(X) \)
  - Propensity model: \( e(X) \)

---
## 3. Machine Learning Models

### Linear Models
- OLS
- Lasso
- Ridge Regression
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

## 4. High-Dimensional Discrete DGP

Covariates are group-structured:

$$
X \in \{0,1\}^{n \times G}
$$

Each observation belongs to exactly one group.

### Outcome model:

$$
Y_i = D_i Y_i(1) + (1 - D_i) Y_i(0)
$$

$$
Y_i(0) = \mu_{g(i)} + \varepsilon_i
\quad , \quad
Y_i(1) = \mu_{g(i)} + \theta + \varepsilon_i
$$

### Treatment assignment:

$$
D_i \sim \mathrm{Bernoulli}(p_{g(i)})
$$

where:

- $\mu_g$: group-level baseline effects  
- $p_g$: group-level propensity (confounded via latent factor)

---

## 5. Latent Confounding Structure
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