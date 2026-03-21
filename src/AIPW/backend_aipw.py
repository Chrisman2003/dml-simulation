import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from sklearn import clone
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# =====================================================
# Paths
# =====================================================
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# =====================================================
# 1. AIPW Estimator
# =====================================================
def aipw(Y, D, m0_hat, m1_hat, e_hat, tau_true):
    #e_hat = np.clip(e_hat, 0.05, 0.95)
    psi = (
        D * (Y - m1_hat) / e_hat
        - (1 - D) * (Y - m0_hat) / (1 - e_hat)
        + m1_hat - m0_hat
    )
    tau_hat = psi.mean()
    var_hat = np.mean((psi - tau_hat) ** 2)
    se = np.sqrt(var_hat / len(Y))
    covered = (tau_hat - 1.96 * se <= tau_true <= tau_hat + 1.96 * se)
    
    return tau_hat, se, covered, e_hat


# =====================================================
# 2. Cross-Fitting Nuisance Estimation
# =====================================================
def cross_fit_nuisances_fast(X, D, Y, learner, K=2):
    n = X.shape[0]
    m0_hat = np.zeros(n)
    m1_hat = np.zeros(n)
    e_hat = np.zeros(n)
    prop_model = LogisticRegression(max_iter=1000)
    kf = KFold(n_splits=K, shuffle=True, random_state=123)

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        D_tr = D[train_idx]
        Y_tr = Y[train_idx]

        # Propensity model
        prop = clone(prop_model)
        prop.fit(X_tr, D_tr)
        e_hat[test_idx] = prop.predict_proba(X_te)[:, 1]

        # Outcome models (already tuned)
        m1 = clone(learner)
        m0 = clone(learner)
        m1.fit(X_tr[D_tr == 1], Y_tr[D_tr == 1])
        m0.fit(X_tr[D_tr == 0], Y_tr[D_tr == 0])

        m1_hat[test_idx] = m1.predict(X_te)
        m0_hat[test_idx] = m0.predict(X_te)

    return m0_hat, m1_hat, e_hat


# =====================================================
# 3. Parallelized Monte Carlo Simulation
# =====================================================
def run_single_sim(s, dgp, learners, n, n_groups, beta_g, p_g):
    rng = np.random.default_rng(s)
    X, D, Y, tau_true = dgp(rng, n, n_groups, beta_g, p_g)
    rows = []
    for name, learner in learners.items():
        m0, m1, e = cross_fit_nuisances_fast(X, D, Y, learner)
        tau, se, cov, e_hat = aipw(Y, D, m0, m1, e, tau_true)
        rows.append({
            "Learner": name,
            "tau": tau,
            "se": se,
            "covered": cov,
            "e_mean": e_hat.mean(),
            "e_var": e_hat.var()
        })
    return rows

def monte_carlo_parallel(dgp, learners, n, sims, n_groups, beta_g, p_g):
    results = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(run_single_sim)(s, dgp, learners, n, n_groups, beta_g, p_g)
        for s in range(sims)
    )
    rows = [item for sublist in results for item in sublist]
    df = pd.DataFrame(rows)
    tau_true = 1.0

    return df.groupby("Learner").agg(
        Mean=("tau", "mean"),
        Bias=("tau", lambda x: x.mean() - tau_true),
        Variance=("tau", "var"),
        Mean_SD_Err=("se", "mean"),
        RMSE=("tau", lambda x: np.sqrt(np.mean((x - tau_true)**2))),
        CI_Width=("se", lambda x: 2 * 1.96 * np.mean(x)),
        Coverage=("covered", "mean"),
    ).reindex(learners.keys())
    

# =====================================================
# 4. Parallelized Hyperparameter Tuning
# =====================================================
# ---------------- Hyperparameter grids ----------------
RIDGE_GRID = {"alpha": np.logspace(-4, 4, 15)}  # 0.001 → 1000
LASSO_GRID = {"alpha": np.logspace(-4, 1, 20)}
EN_GRID = {"alpha": np.logspace(-4, 1, 15), "l1_ratio": np.linspace(0.1, 0.9, 9)}
#LOGIT_LASSO_GRID = {"C": np.logspace(-3, 2, 25)}
RF_GRID = {"max_depth": [3, 5, 10, None], "min_samples_leaf": [1, 5, 10, 20], "max_features": ['sqrt', 'log2', None]}
GB_GRID = {"learning_rate": [0.01, 0.05, 0.1], "max_depth": [2, 3, 5], "n_estimators": [100, 300, 500], "subsample": [0.5, 0.8, 1.0]} 
CATBOOST_GRID = {"learning_rate": [0.01, 0.05, 0.1], "depth": [3, 5, 7], "iterations": [100, 300, 500]}
XGB_GRID = {"learning_rate": [0.01, 0.05, 0.1], "max_depth": [2, 3, 5, 7], "n_estimators": [100, 300, 500], "subsample": [0.5, 0.8, 1.0], "colsample_bytree": [0.5, 0.8, 1.0]}

# ---------------- Single learner tuner ----------------
def tune_learner(model, param_grid, X, y):
    if param_grid is None:
        model.fit(X, y)
        return model
    gs = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=1)
    gs.fit(X, y)
    return gs.best_estimator_

# ---------------- Handle each learner ----------------
def tune_single(name, model, X, Y):
    if name == "Ridge":
        grid = RIDGE_GRID
    elif name == "Lasso":
        grid = LASSO_GRID
    elif name == "ElasticNet":
        grid = EN_GRID
#    elif name == "LogitLasso":
#        grid = LOGIT_LASSO_GRID
    elif name == "RF":
        grid = RF_GRID
    elif name == "GB":
        grid = GB_GRID
    elif name == "CatBoost":
        grid = CATBOOST_GRID
    elif name == "XGBoost":
        grid = XGB_GRID
    else:
        return name, clone(model)
    
    tuned_model = tune_learner(clone(model), grid, X, Y)
    return name, tuned_model

# ---------------- Parallel tuning ----------------
def tune_once_parallel(dgp_func, learners, n, n_groups, beta_g, p_g, seed=0):
    rng = np.random.default_rng(seed)
    X, D, Y, _ = dgp_func(rng, n, n_groups, beta_g, p_g)
    results = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(tune_single)(name, model, X, Y)
        for name, model in learners.items()
    )
    tuned = dict(results)
    return tuned