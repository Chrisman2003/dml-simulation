import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =====================================================
# Paths
# =====================================================
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# =====================================================
# 1. Data Generating Process
# =====================================================
def generate_aipw_data(n, p, rng, tau=1.0, sigma=1.0):
    """
    High-dimensional DGP with binary treatment and known ATE.
    """
    X = rng.binomial(1, 0.5, size=(n, p))
    D = rng.binomial(1, 0.5, size=n)

    beta = np.zeros(p)
    beta[1:10] = 0.5  # nuisance covariates

    Y = tau * D + X @ beta + rng.normal(0, sigma, size=n)
    return X, D, Y

# =====================================================
# 2. AIPW Estimator
# =====================================================
def aipw_ate(Y, D, e_hat, m0_hat, m1_hat):
    """
    Augmented Inverse Probability Weighting estimator.
    """
    eps = 1e-8
    e_hat = np.clip(e_hat, eps, 1 - eps)
    score = (
        D * (Y - m1_hat) / e_hat
        - (1 - D) * (Y - m0_hat) / (1 - e_hat)
        + (m1_hat - m0_hat)
    )
    return np.mean(score)

# =====================================================
# 3. One Monte Carlo Draw
# =====================================================
def run_single_simulation(seed, n, p, tau):

    rng = np.random.default_rng(seed)
    X, D, Y = generate_aipw_data(n, p, rng, tau=tau)

    # -----------------------------
    # Propensity score model
    # -----------------------------
    prop_model = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(max_iter=1000))
    ])
    prop_model.fit(X, D)
    e_hat = prop_model.predict_proba(X)[:, 1]

    # -----------------------------
    # Outcome regression models
    # -----------------------------
    outcome_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=0.06, max_iter=5000))
    ])

    outcome_model.fit(X[D == 1], Y[D == 1])
    m1_hat = outcome_model.predict(X)

    outcome_model.fit(X[D == 0], Y[D == 0])
    m0_hat = outcome_model.predict(X)

    # -----------------------------
    # AIPW estimate
    # -----------------------------
    tau_hat = aipw_ate(Y, D, e_hat, m0_hat, m1_hat)

    return tau_hat

# =====================================================
# 4. Main Monte Carlo
# =====================================================
if __name__ == "__main__":

    # -----------------------------
    # Parameters
    # -----------------------------
    n = 300
    p = 500
    tau_true = 1.0
    n_sims = 1000

    print(f"Running AIPW Monte Carlo (n={n}, p={p}, sims={n_sims})")

    estimates = Parallel(n_jobs=-1)(
        delayed(run_single_simulation)(i, n, p, tau_true)
        for i in range(n_sims)
    )

    estimates = np.array(estimates)

    # -----------------------------
    # Summary statistics
    # -----------------------------
    summary = pd.DataFrame({
        "Mean Estimate": [estimates.mean()],
        "Bias": [estimates.mean() - tau_true],
        "Std Dev": [estimates.std()]
    })

    print("\nAIPW Monte Carlo Summary")
    print("------------------------")
    print(summary.to_string(index=False))