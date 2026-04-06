import os
import numpy as np
import pandas as pd
import cvxpy as cp # New Library
import statsmodels.api as sm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pathlib import Path
from sklearn import clone
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.linalg import lstsq
from sklearn.base import BaseEstimator, RegressorMixin
# =====================================================
# Paths
# =====================================================
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
# =====================================================
# 1. AIPW Estimator
# =====================================================
def aipw(Y, D, m0_hat, m1_hat, e_hat, tau_true):
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
def cross_fit_nuisances_fast(X, D, Y, learner, K=5):
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
        
        ## DEBUG CHECK
        #print(f"Fold {len(train_idx)}: Treated={D_tr.sum()}, Untreated={len(D_tr) - D_tr.sum()}") # Print sample size of Treated and Untreated
        #X_sub = X_tr[D_tr == 1]
        #active_cols = np.sum(X_sub, axis=0) > 0
        #print("Active features:", active_cols.sum(), "out of", X_sub.shape[1])

        # Propensity model
        prop = clone(prop_model)
        prop.fit(X_tr, D_tr)
        e_hat[test_idx] = prop.predict_proba(X_te)[:, 1]

        # Outcome models (already tuned)
        m1 = clone(learner)
        m0 = clone(learner)
        m1.fit(X_tr[D_tr == 1], Y_tr[D_tr == 1]) # Filters Rows but not Columns
        m0.fit(X_tr[D_tr == 0], Y_tr[D_tr == 0]) # Filters Rows but not Columns
        
        m1_hat[test_idx] = m1.predict(X_te).ravel()
        m0_hat[test_idx] = m0.predict(X_te).ravel()

    return m0_hat, m1_hat, e_hat


# =====================================================
# 3. Parallelized Monte Carlo Simulation
# =====================================================
def run_single_sim(s, dgp, learners, n, n_groups, beta_g, p_g):
    #print("Current Simulation", s)
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
RF_GRID = {"max_depth": [3, 5, 10, None], "min_samples_leaf": [1, 5, 10, 20], "max_features": ['sqrt', 'log2', None]}
GB_GRID = {"learning_rate": [0.01, 0.05, 0.1], "max_depth": [2, 3, 5], "n_estimators": [100, 300, 500], "subsample": [0.5, 0.8, 1.0]} 
CATBOOST_GRID = {"learning_rate": [0.01, 0.05, 0.1], "depth": [3, 5, 7], "iterations": [100, 300, 500]}
XGB_GRID = {"learning_rate": [0.01, 0.05, 0.1], "max_depth": [2, 3, 5, 7], "n_estimators": [100, 300, 500], "subsample": [0.5, 0.8, 1.0], "colsample_bytree": [0.5, 0.8, 1.0]}
KRR_GRID = {"alpha": np.logspace(-4, 2, 10), "gamma": np.logspace(-4, 2, 10)}
SVR_GRID = {"C": np.logspace(-2, 3, 10), "gamma": np.logspace(-4, 1, 10), "epsilon": [0.01, 0.1, 0.5]}
FUSED_GRID = {"alpha1": np.logspace(-3, 0, 4), "alpha2": np.logspace(-3, 1, 4)}  # Sparsity + Fusion jumps
EXTRATREES_GRID = {"n_estimators": [100, 300, 500], "max_depth": [3, 5, 10, None], "min_samples_leaf": [1, 5, 10, 20], "max_features": ['sqrt', 'log2', None],"bootstrap": [True]}
HONEST_FOREST_GRID = {"n_estimators": [100, 300], "max_depth": [5, 10, None], "min_samples_leaf": [5, 10, 20], "max_features": [0.3, 0.5, 'sqrt'], "min_impurity_decrease": [0.0, 1e-4]}
#KNN_GRID = {"n_neighbors": [3, 5, 10, 20, 50], "weights": ['uniform', 'distance'], "metric": ['euclidean', 'manhattan', 'minkowski'], "p": [1, 2]}
#PCR_GRID = {"pca__n_components": [2, 5, 10, 20]}
#NN_GRID = {"hidden_layer_sizes": [(10, ), (20, ), (10, 5)], "activation": ["relu", "tanh"], "alpha": np.logspace(-1, 3, 5), "learning_rate_init": [0.001, 0.01], "max_iter": [2000], "early_stopping": [False]}
NN_GRID = {
    "mlp__solver": ["lbfgs"],
    "mlp__hidden_layer_sizes": [(10,), (20,), (10, 5)], 
    "mlp__activation": ["relu", "tanh"], 
    "mlp__alpha": np.logspace(-1, 3, 5), 
    "mlp__learning_rate_init": [0.001, 0.01], 
    "mlp__max_iter": [2000], 
    "mlp__early_stopping": [False]
}
# --> Early Stopping True Parameter catastrophic for small sample sizes

# ---------------- Single learner tuner ----------------
def tune_learner(model, param_grid, X, y):
    if param_grid is None:
        model.fit(X, y)
        return model
    gs = GridSearchCV(model, param_grid, cv=2, scoring="neg_mean_squared_error", n_jobs=1)
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
    elif name == "RF":
        grid = RF_GRID
    elif name == "GB":
        grid = GB_GRID
    elif name == "CatBoost":
        grid = CATBOOST_GRID
    elif name == "XGBoost":
        grid = XGB_GRID
    elif name == "KRR_RBF":
        grid = KRR_GRID
    elif name == "SVR_RBF":
        grid = SVR_GRID
    elif name == "FusedLasso":
        grid = FUSED_GRID
    elif name == "ExtraTrees":
        grid = EXTRATREES_GRID
    elif name == "HonestForest":
        grid = HONEST_FOREST_GRID
    #elif name == "KNN":
    #    grid = {f"knn__{k}": v for k, v in KNN_GRID.items()}
    #elif name == "PCR":
    #    n_features = X.shape[1]
    #    components = [c for c in [2, 5, 10, 20] if c <= n_features]
    #    grid = {"pca__n_components": components}
    elif name == "NeuralNet":
        grid = NN_GRID
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


# =====================================================
# 5. Configured Estimators
# =====================================================
class FusedLasso(BaseEstimator, RegressorMixin):
    """
    Custom Sklearn-compatible Fused Lasso using CVXPY.
    """
    def __init__(self, alpha1=1.0, alpha2=1.0):
        self.alpha1 = alpha1  # Standard Lasso Penalty (Sparsity)
        self.alpha2 = alpha2  # Total Variation Penalty (Fusion)
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        n, p = X.shape
        
        # Define CVXPY variables
        beta = cp.Variable(p)
        intercept = cp.Variable()
        
        # 1. Data Fidelity: (1 / 2n) * Sum of Squared Errors
        rss = (1.0 / (2 * n)) * cp.sum_squares(y - X @ beta - intercept)
        
        # 2. Penalty Mechanisms
        # alpha1 controls standard sparsity (forcing betas to 0)
        l1_penalty = self.alpha1 * cp.norm1(beta)
        
        # alpha2 controls the "Fusion" (forcing adjacent betas to be equal)
        fused_penalty = self.alpha2 * cp.norm1(cp.diff(beta))
        
        # Objective function
        objective = cp.Minimize(rss + l1_penalty + fused_penalty)
        problem = cp.Problem(objective)
        
        # Solve using ECOS or SCS (robust open-source solvers inside CVXPY)
        problem.solve(solver=cp.SCS) 
        
        # Extract values
        self.coef_ = beta.value
        self.intercept_ = intercept.value
        
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
    

class StatsmodelsOLS(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # We add the constant here to get p+1 columns
        X_ = sm.add_constant(X, has_constant='add') if self.fit_intercept else X
        self.model_ = sm.OLS(y, X_).fit()
        return self

    def predict(self, X):
        # We MUST add the constant here too, or the shapes won't align!
        X_ = sm.add_constant(X, has_constant='add') if self.fit_intercept else X
        return self.model_.predict(X_)


class NumpyOLS(BaseEstimator, RegressorMixin):
    def __init__(self):
        # no hyperparameters → empty init is fine
        pass

    def fit(self, X, y):
        X_ = np.column_stack([np.ones(X.shape[0]), X])
        self.beta_, *_ = np.linalg.lstsq(X_, y, rcond=None)
        return self

    def predict(self, X):
        X_ = np.column_stack([np.ones(X.shape[0]), X])
        return X_ @ self.beta_
    
    
class ScipyOLS(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        X_ = np.column_stack([np.ones(X.shape[0]), X])
        self.beta_, *_ = lstsq(X_, y)
        return self

    def predict(self, X):
        X_ = np.column_stack([np.ones(X.shape[0]), X])
        return X_ @ self.beta_
    
    
class Optimized_CVX_OLS(BaseEstimator, RegressorMixin):
    """
    OLS Estimator that avoids Moore-Penrose pseudo-inverse.
    Solves the objective function: min ||y - Xb||^2 via convex optimization.
    Uses CVX
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n, p = X.shape
        
        # Define variables
        beta = cp.Variable(p)
        intercept = cp.Variable() if self.fit_intercept else 0
        
        # Objective: Minimize Sum of Squared Residuals (SSR)
        # cp.sum_squares is equivalent to ||y - (Xb + c)||^2
        objective = cp.Minimize(cp.sum_squares(y - X @ beta - intercept))
        
        # Define and solve problem
        # We use SCS or ECOS; these don't rely on the Moore-Penrose matrix
        prob = cp.Problem(objective)
        prob.solve(solver=cp.SCS) 
        
        # Extract results
        self.coef_ = beta.value
        if self.fit_intercept:
            self.intercept_ = intercept.value
        else:
            self.intercept_ = 0.0
            
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
    
    
class StrictNormalOLS(BaseEstimator, RegressorMixin):
    """
    A 'Textbook' OLS using the Normal Equation.
    This will fail or produce 'Singular Matrix' errors if p > n.
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        nonzero_cols = np.any(X != 0, axis=0)
        X_reduced = X[:, nonzero_cols]
        if self.fit_intercept:
            X_full = np.column_stack([np.ones(X_reduced.shape[0]), X_reduced])
        else:
            X_full = X_reduced

        xtx = X_full.T @ X_full
        xtx_inv = np.linalg.inv(xtx)
        beta_full = xtx_inv @ (X_full.T @ y)
        self.nonzero_cols_ = nonzero_cols
        if self.fit_intercept:
            self.intercept_ = beta_full[0]
            self.coef_ = beta_full[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta_full
        return self

    def predict(self, X):
        X_reduced = X[:, self.nonzero_cols_]
        return X_reduced @ self.coef_ + self.intercept_

    #def fit(self, X, y):
    #    # 1. Add constant for intercept
    #    if self.fit_intercept:
    #        X_full = np.column_stack([np.ones(X.shape[0]), X])
    #    else:
    #        X_full = X
    #    
    #    # 2. Compute (X'X)
    #    xtx = X_full.T @ X_full
    #    
    #    # 3. Try to invert (X'X) - This is where it fails if p > n
    #    # We use np.linalg.inv instead of pinv (pseudo-inverse)
    #    try:
    #        xtx_inv = np.linalg.inv(xtx) # Function: np.linalg.pinv() computes (Moore-Penrose) pseudo-inverse of Matrix
    #    except np.linalg.LinAlgError:
    #        raise ValueError("OLS Failed: Matrix is singular (p > n or perfect multicollinearity).")
    #        
    #    # 4. Solve for beta: beta = (X'X)^-1 X'y
    #    beta_full = xtx_inv @ (X_full.T @ y)
    #    
    #    if self.fit_intercept:
    #        self.intercept_ = beta_full[0]
    #        self.coef_ = beta_full[1:]
    #    else:
    #        self.intercept_ = 0.0
    #        self.coef_ = beta_full
    #        
    #    return self

    #def predict(self, X):
    #    return X @ self.coef_ + self.intercept_
    
# CHAT-GPT Fix
class SafeNormalOLS(BaseEstimator, RegressorMixin):
    """
    Computes OLS using the Normal Equation but dynamically detects and drops 
    empty columns (groups with no treated/untreated units in this specific fold),
    preventing the 'Singular Matrix' divide-by-zero crash.
    """
    def __init__(self, fit_intercept=False): # Changed default to False to avoid Dummy Trap
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # 1. Add intercept if requested
        if self.fit_intercept:
            X_full = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_full = X
            
        # 2. IDENTIFY EMPTY COLUMNS: Find columns that have actual variance/data
        # If a column sum is 0, no one in this subset belongs to that group
        active_cols = np.abs(X_full).sum(axis=0) > 1e-10
        
        # 3. Subset the matrix to ONLY active columns
        X_active = X_full[:, active_cols]
        
        # 4. Compute (X'X) for active columns
        xtx = X_active.T @ X_active
        
        try:
            xtx_inv = np.linalg.inv(xtx)
        except np.linalg.LinAlgError:
            raise ValueError("OLS Failed: Dummy Variable Trap or Perfect Multicollinearity detected.")
            
        # 5. Solve for the active coefficients
        beta_active = xtx_inv @ (X_active.T @ y)
        
        # 6. Reconstruct the full beta vector (assigning 0 to the empty groups)
        beta_full = np.zeros(X_full.shape[1])
        beta_full[active_cols] = beta_active
        
        # 7. Store results
        if self.fit_intercept:
            self.intercept_ = beta_full[0]
            self.coef_ = beta_full[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta_full
            
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
    
# =====================================================
# 6. Plotting Logic
# =====================================================
def plot_metrics_vs_x(df, x_col, title_suffix, output_dir, filename_prefix):
    metrics = ["Coverage", "RMSE", "CI_Width", "Mean", "Bias", "Variance"]
    figures = {}
    
    # CONSISTENT COLOR MAP
    unique_learners = df["Learner"].unique()
    colors = list(plt.cm.tab10.colors) + list(plt.cm.Dark2.colors)
    learner_color_map = dict(zip(unique_learners, colors))

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        for learner in unique_learners:
            subset = df[df["Learner"] == learner].sort_values(x_col)
            ax.plot(
                subset[x_col],
                subset[metric],
                marker="o",
                label=learner,
                color=learner_color_map[learner],
            )
        if metric == "Coverage":
            ax.axhline(y=0.95, linestyle="--", color="red")
        
        ax.set_title(f"{metric} vs {x_col} ({title_suffix})")
        ax.set_xlabel(x_col)
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()

        figures[metric] = fig
        save_path = os.path.join(output_dir, metric)
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(save_path, f"{filename_prefix}.png"),#_{metric}_{x_col}.png"),
            dpi=500,
            bbox_inches="tight",
        )
    return figures

def save_results_table(df, output_dir, filename_prefix):
    os.makedirs(output_dir, exist_ok=True)
    df = df.round(4)
    df.to_csv(os.path.join(output_dir, f"{filename_prefix}.csv")) # Save Raw CSV
    
    return df