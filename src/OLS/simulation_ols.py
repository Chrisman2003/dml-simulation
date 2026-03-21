"""
Monte Carlo Simulation Study with a normal distribution
Principles being demonstrated:
1) Law of Large Numbers
2) Central Limit Theorem
3) Confidence Intervals
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ----------------------------
# Data-generating process
# ----------------------------
def sample_normal(mu, sigma, n, rng):
    return rng.normal(loc=mu, scale=sigma, size=n) # scale is std dev

def sample_mean(x):
    return np.mean(x)

def sample_variance(x):
    return np.var(x, ddof=1)

# ----------------------------
# Confidence interval
# ----------------------------
def mean_ci(x, alpha):
    n = len(x)
    mean = sample_mean(x)
    std_error = np.std(x, ddof=1) / np.sqrt(n)
    z = norm.ppf(1 - alpha / 2)
    lower = mean - z * std_error
    upper = mean + z * std_error
    return lower, upper

# ----------------------------
# One Monte Carlo run
# ----------------------------
def run_once(n, mu, sigma, alpha, rng):
    x = sample_normal(mu, sigma, n, rng)
    mean = sample_mean(x)
    var = sample_variance(x)
    ci_lower, ci_upper = mean_ci(x, alpha)
    covered = (ci_lower <= mu) and (mu <= ci_upper)

    return {
        "mean": mean,
        "variance": var,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "covered": covered
    }

# ----------------------------
# Monte Carlo simulation
# ----------------------------
def monte_carlo(n, mu, sigma, alpha, n_rep, seed=42):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_rep):
        output = run_once(n, mu, sigma, alpha, rng)
        results.append(output)
    return results

# ----------------------------
# Diagnostics
# ----------------------------
def summarize_results(results, mu):
    means = np.array([r["mean"] for r in results])
    coverage = np.mean([r["covered"] for r in results])

    return {
        "mean_of_means": means.mean(),
        "bias": means.mean() - mu,
        "variance": means.var(),
        "coverage": coverage
    }

# ----------------------------
# Plots
# ----------------------------
def plot_sampling_distribution(results, mu, save=True):
    means = [r["mean"] for r in results]
    
    plt.figure(figsize=(8, 5))
    plt.hist(means, bins=30, density=True, alpha=0.7)
    plt.axvline(mu, color="red", linestyle="--", label="True mean")
    plt.title("Sampling Distribution of Sample Mean")
    plt.xlabel("Sample mean")
    plt.legend()
    plt.savefig(FIGURES_DIR / "sampling_distribution.png", dpi=300)

def plot_confidence_intervals(results, mu, max_plots=100, save=True):
    plt.figure(figsize=(8, 5))
    
    for i, r in enumerate(results[:max_plots]):
        plt.plot([i, i], [r["ci_lower"], r["ci_upper"]], color="gray")
        plt.plot(i, r["mean"], "bo", markersize=3)
    
    plt.axhline(mu, color="red", linestyle="--", label="True mean")
    plt.title("Confidence Intervals (First 100 Replications)")
    plt.xlabel("Replication")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(FIGURES_DIR / "confidence_intervals.png", dpi=300)

def convergence_experiment(ns, mu, sigma, alpha, n_rep, save=True):
    avg_means = []
    for n in ns:
        results = monte_carlo(n, mu, sigma, alpha, n_rep)
        avg_means.append(np.mean([r["mean"] for r in results]))

    plt.figure(figsize=(8, 5))
    plt.plot(ns, avg_means, marker="o")
    plt.axhline(mu, color="red", linestyle="--", label="True mean")
    plt.xscale("log")
    plt.title("Convergence of Sample Mean (LLN)")
    plt.xlabel("Sample size n (log scale)")
    plt.ylabel("Average sample mean")
    plt.legend()
    plt.savefig(FIGURES_DIR / "convergence.png", dpi=300)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":

    mu = 1.0 # True mean
    sigma = 2.0 # True standard deviation
    alpha = 0.05
    n = 30
    n_rep = 1000

    results = monte_carlo(n, mu, sigma, alpha, n_rep)

    summary = summarize_results(results, mu)
    print("Monte Carlo Summary")
    print("-------------------")
    for k, v in summary.items():
        print(f"{k}: {v}")

    plot_sampling_distribution(results, mu)
    plot_confidence_intervals(results, mu)

    # Convergence
    ns = [10, 30, 100, 300, 1000, 3000]
    convergence_experiment(ns, mu, sigma, alpha, n_rep=500)