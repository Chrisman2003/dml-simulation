import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =====================================================
# Paths
# =====================================================
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# =====================================================
# 1. Data Generating Process (Population)
# =====================================================
def generate_dataset_with_known_beta(n, p, rng, beta0=1.0, sigma=1.0):
    X = rng.binomial(1, 0.5, size=(n, p))

    beta = np.zeros(p)
    beta[0] = beta0
    beta[1:10] = 0.5

    Y = X @ beta + rng.normal(0, sigma, size=n)
    return X, Y

# =====================================================
# 2. Partial Effect Estimation (Estimators)
# =====================================================
def estimate_partial_effect(model, X, Y):
    """
    Generic marginal effect estimator:
    E[ f(X | X0=1) - f(X | X0=0) ]
    """
    model.fit(X, Y)

    X1 = X.copy()
    X0 = X.copy()
    X1[:, 0] = 1
    X0[:, 0] = 0

    return np.mean(model.predict(X1) - model.predict(X0))

# =====================================================
# 3. One Monte Carlo Draw
# =====================================================
def run_single_simulation(seed, n, p, learners):
    rng = np.random.default_rng(seed)
    X, Y = generate_dataset_with_known_beta(n, p, rng)

    estimates = {}
    for name, model in learners.items():
        estimates[name] = estimate_partial_effect(model, X, Y)

    return estimates

# =====================================================
# 4. Main Monte Carlo
# =====================================================
if __name__ == "__main__":

    # -----------------------------
    # Parameters
    # -----------------------------
    n, p = 300, 500
    n_sims = 1000
    true_beta = 1.0

    # -----------------------------
    # Learners (best params from CV)
    # -----------------------------
    learners = {
        "OLS": LinearRegression(),

        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.05995, max_iter=5000))
        ]),

        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.21544, l1_ratio=0.5, max_iter=5000))
        ]),

        "RandomForest": RandomForestRegressor(
            max_depth=10,
            min_samples_leaf=5,
            random_state=0
        ),

        "GBoost": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=2,
            random_state=0
        )
    }

    print(f"Running {n_sims} Monte Carlo simulations...")

    # -----------------------------
    # Parallel Monte Carlo
    # -----------------------------
    results = Parallel(n_jobs=-1)(
        delayed(run_single_simulation)(i, n, p, learners)
        for i in range(n_sims)
    )

    df = pd.DataFrame(results)

    # =====================================================
    # 5. Bias Summary
    # =====================================================
    summary = pd.DataFrame({
        "Mean Estimate": df.mean(),
        "Bias": df.mean() - true_beta,
        "Std Dev": df.std()
    })

    print("\nBias Summary (β₀ = 1.0):")
    print(summary)

    # =====================================================
    # 6. Plots
    # =====================================================
    for learner in df.columns:
        plt.figure(figsize=(7, 4))
        plt.hist(df[learner], bins=30, alpha=0.7, edgecolor="black")
        plt.axvline(true_beta, color="red", linestyle="--", label="True β₀")
        plt.axvline(df[learner].mean(), color="green", label="Mean Estimate")
        plt.title(f"Sampling Distribution – {learner}")
        plt.xlabel("Estimated Effect")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"sampling_{learner}.png", dpi=300)
        plt.close()

    print("\nFigures saved in ./figures/")