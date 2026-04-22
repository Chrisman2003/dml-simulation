# Debiased Machine Learning in High-Dimensional Discrete Settings

This project studies **Double / Debiased Machine Learning (DML)** in high-dimensional settings with **discrete and grouped covariates**, focusing on how nuisance function estimators behave under varying sparsity, dimensionality, sample size, and confounding structures.

We implement and evaluate the **Augmented Inverse Probability Weighting (AIPW)** estimator with **cross-fitted machine learning nuisance models**, and systematically compare its performance under these machine learners with synthetic data-generating processes.

## Configuration
- Run in Terminal the command "pip install -e ."

## Structure
- A generalized file structure: frontends, backends, (middleend for code serving as a conceptual bridge) is used
- The core pipeline is under "src/AIPW"
- In "src/AIPW/frontend_aipw.ipynb", the Machine Learners utilized for nuisance estimation are configurable in the set "learners_regime". A maximum of 26 Learners can be used for the simulation runs, with the configured colour palette in "backend_aipw.py"

---

# Research Objective

We investigate how nuisance estimation impacts causal inference under the following conditions:

- Moderate to High-dimensional regimes
- Group-structured covariates (one-hot encoded)
- Latent confounding in treatment assignment
- Sparse vs dense signal regimes

# Methodology

## 1. AIPW Estimator

The **Augmented Inverse Probability Weighting (AIPW)** is defined as:

$$
\hat{\theta}_{\text{AIPW}} = \frac{1}{n} \sum_{i=1}^{n}\left[\frac{D_i (Y_i - m_1(X_i))}{e(X_i)} - \frac{(1 - D_i)(Y_i - m_0(X_i))}{1 - e(X_i)}+ m_1(X_i) - m_0(X_i)\right]
$$

---

## 2. Sample-Fitting Procedure
- **Cross-Fitting (nuisance estimation):**
  - K-fold sample splitting
  - Outcome models: $m_0(X), m_1(X)$
  - Propensity model: $e(X)$
  - Relaxes the class of utilizable Machine Learners

- **Cross-Validation (hyperparameter tuning):**
  - Performed separately within training data
  - GridSearchCV used to select optimal model parameters
  - Applied across all learners before entering the DML pipeline

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
D_i \sim \mathrm{Bernoulli}(\pi_{g(i)})
$$

where:

- $\mu_g$: group-level baseline effects  
- $\pi_g$: group-level propensity (confounded via latent factor)

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
\pi_g = \sigma(\rho_g)
$$

where $\sigma(\cdot)$ denotes the logistic function.
In the thesis notation (unlike the code implementation), the quantity $\pi_g$ is denoted as $\pi_g$ to maintain a consistent formal probabilistic notation for propensity scores throughout the text.

This induces **strong correlation between treatment and outcome mechanisms**.