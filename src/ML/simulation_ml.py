import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ----------------------------
# Paths
# ----------------------------
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ----------------------------
# Data-generating process
# ----------------------------
def generate_single_dataset(n, p, sigma, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.binomial(1, 0.5, size=(n, p))
    beta = rng.normal(0, 1, size=p)
    beta[10:] = 0  # Sparse signal: only 10 covariates
    eps = rng.normal(0, sigma, size=n)
    Y = X @ beta + eps
    return X, Y

# ----------------------------
# Evaluation Function
# ----------------------------
def evaluate_learners(X, Y, n_folds=5):
    """
    Runs GridSearchCV once for each learner and returns 
    the best out-of-sample performance.
    """
    results = []

    # 1. OLS (Baseline)
    ols_pipe = Pipeline([("model", LinearRegression())])
    # No params to tune for OLS, but we use GridSearch for consistent scoring
    ols_grid = GridSearchCV(ols_pipe, {}, cv=n_folds, scoring="neg_mean_squared_error")
    ols_grid.fit(X, Y)
    results.append({"Learner": "OLS", "MSE": -ols_grid.best_score_, "Params": "None"})

    # 2. LASSO
    lasso_pipe = Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=5000))])
    lasso_params = {"model__alpha": np.logspace(-4, 1, 10)}
    lasso_grid = GridSearchCV(lasso_pipe, lasso_params, cv=n_folds, scoring="neg_mean_squared_error")
    lasso_grid.fit(X, Y)
    results.append({"Learner": "Lasso", "MSE": -lasso_grid.best_score_, "Params": lasso_grid.best_params_})

    # 3. Elasticnet
    en_pipe = Pipeline([
        ("scaler", StandardScaler()), 
        ("model", ElasticNet(max_iter=5000))
    ])
    en_params = {
        "model__alpha": np.logspace(-4, 1, 10),
        "model__l1_ratio": [0.1, 0.5, 0.7, 0.9]
    }
    en_grid = GridSearchCV(en_pipe, en_params, cv=n_folds, scoring="neg_mean_squared_error", n_jobs=-1)
    en_grid.fit(X, Y)
    results.append({
        "Learner": "ElasticNet", 
        "MSE": -en_grid.best_score_, 
        "Params": en_grid.best_params_
    })

    # 4. Random Forest
    rf_pipe = Pipeline([("model", RandomForestRegressor(random_state=42))])
    rf_params = {
        "model__max_depth": [3, 5, 10], 
        "model__min_samples_leaf": [1, 5]
    }
    rf_grid = GridSearchCV(rf_pipe, rf_params, cv=n_folds, scoring="neg_mean_squared_error")
    rf_grid.fit(X, Y)
    results.append({"Learner": "RandomForest", "MSE": -rf_grid.best_score_, "Params": rf_grid.best_params_})

    # 5. Gradient Boosting
    gb_pipe = Pipeline([("model", GradientBoostingRegressor(random_state=42))])
    gb_params = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.01, 0.1],
        "model__max_depth": [2, 3]
    }
    gb_grid = GridSearchCV(gb_pipe, gb_params, cv=n_folds, scoring="neg_mean_squared_error")
    gb_grid.fit(X, Y)
    results.append({"Learner": "GBoost", "MSE": -gb_grid.best_score_, "Params": gb_grid.best_params_})

    return pd.DataFrame(results)

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # Parameters
    n = 300
    p = 500  # High-dimensional case p > n
    sigma = 1.0

    print(f"Generating data (n={n}, p={p})...")
    X, Y = generate_single_dataset(n, p, sigma)

    print("Running Cross-Validation for Hyperparameter Tuning...")
    df_results = evaluate_learners(X, Y)

    # Print Results Table
    print("\nComparison of Learners (Out-of-sample MSE):")
    print(df_results[["Learner", "MSE"]].to_string(index=False))

    # Save visualization
    plt.figure(figsize=(10, 6))
    plt.bar(df_results["Learner"], df_results["MSE"], color='skyblue', edgecolor='black')
    plt.ylabel("Cross-Validated MSE")
    plt.title(f"Model Comparison (p={p}, n={n})")
    plt.savefig(FIGURES_DIR / "learner_comparison_cv.png", dpi=300)
    
    print(f"\nBest parameters for OLS: {df_results.iloc[0]['Params']}")
    print(f"\nBest parameters for Lasso: {df_results.iloc[1]['Params']}") # Best alpha for Lasso
    print(f"\nBest parameters for ElasticNet: {df_results.iloc[2]['Params']}") # Best alpha and l1_ratio for ElasticNet
    print(f"\nBest parameters for Random Forest: {df_results.iloc[3]['Params']}") # Best max_depth and min_samples_leaf for Random Forest
    print(f"\nBest parameters for Gradient Boosting: {df_results.iloc[4]['Params']}") # Best n_estimators, learning_rate, max_depth for Gradient Boosting
    
    