from zipfile import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from joblib import Parallel, delayed
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ----------------------------
# 1. Define the Ground Truth
# ----------------------------
def generate_dataset_with_known_beta(n, p, rng, true_beta_val=1.0):
    """
    Creates a dataset where we explicitly set the coefficient 
    of the first variable to a known value.
    """
    X = rng.binomial(1, 0.5, size=(n, p))
    
    # We define a sparse coefficient vector
    beta = np.zeros(p)
    beta[0] = true_beta_val  # This is the "Mean" we want to recover
    beta[1:10] = 0.5 # Add some other small effects (nuisance)
    
    # Generate Y with some noise (sigma=1.0)
    Y = X @ beta + rng.normal(0, 1.0, size=n)
    return X, Y

# ----------------------------
# 2. The Estimation Step
# ----------------------------
def run_parameter_simulation(n, p, seed, best_alpha):
    """
    Runs one single estimation using the 'Best Alpha' 
    found in your previous calibration script.
    """
    rng = np.random.default_rng(seed)
    X, Y = generate_dataset_with_known_beta(n, p, rng, true_beta_val=1.0)
    
    # Use the best alpha identified in Script 2
    model = Lasso(alpha=best_alpha)
    model.fit(X, Y)
    
    # We return the estimate of the first coefficient
    return model.coef_[0]

# ----------------------------
# 3. The Parallel Monte Carlo
# ----------------------------
if __name__ == "__main__":
    # Parameters from your previous output
    n, p = 300, 500
    best_alpha = 0.06  # Take this from your Script 2 results
    n_sims = 1000      # 
    
    print(f"Running {n_sims} simulations in parallel...")

    # n_jobs=-1 uses all available CPU cores
    estimates = Parallel(n_jobs=-1)(
        delayed(run_parameter_simulation)(n, p, i, best_alpha) 
        for i in range(n_sims)
    )

    # ----------------------------
    # 4. Analyze the Results
    # ----------------------------
    avg_estimate = np.mean(estimates)
    bias = avg_estimate - 1.0  # (Average - True Value)
    
    print(f"True Beta: 1.0")
    print(f"Average ML Estimate: {avg_estimate:.4f}")
    print(f"Bias: {bias:.4f}")

    # Plotting the distribution of the 'Mean'
    plt.figure(figsize=(8, 5))
    plt.hist(estimates, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(1.0, color='red', linestyle='--', label='True Value (1.0)')
    plt.axvline(avg_estimate, color='green', linestyle='-', label=f'Mean Estimate ({avg_estimate:.2f})')
    plt.title(f"Sampling Distribution of Beta_0 (n={n}, p={p})")
    plt.xlabel("Estimated Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(FIGURES_DIR / "beta_estimate_distribution.png", dpi=300)