import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("results/Tables/NewVaryingSPARS.csv")

class_map = {
    # Linear-based
    "OLS_sklearn": "Linear-based",
    #"OLS_Strict": "Linear-based",
    #"OLS_Safe": "Linear-based",
    "Lasso": "Linear-based",
    "Ridge": "Linear-based",
    "ElasticNet": "Linear-based",
    "FusedLasso": "Linear-based",

    # Tree-based
    "RF": "Tree-based",
    "ExtraTrees": "Tree-based",
    "HonestForest": "Tree-based",
    
    # Boosted Decision Trees
    "GB": "Boosted Decision Tree-based",
    "CatBoost": "Boosted Decision Tree-based",
    "XGBoost": "Boosted Decision Tree-based",

    # Kernel-based
    "KRR_RBF": "Kernel-based",
    "SVR_RBF": "Kernel-based",

    # Other
    "NeuralNet": "Neural Network-based"
}

# Perform Mapping
df["Class"] = df["Learner"].map(class_map)
# Compute Class Averages
grouped = (
    df.groupby(["Sparsity", "Class"])["Coverage"]
    .mean()
    .reset_index()
)

# Plot Figures
plt.figure()
for cls in grouped["Class"].unique():
    subset = grouped[grouped["Class"] == cls]
    plt.plot(
        subset["Sparsity"],
        subset["Coverage"],
        marker='o',
        label=cls
    )
plt.axhline(y=0.95, linestyle='--')  # nominal coverage
plt.xlabel("Sparsity")
plt.ylabel("Coverage")
plt.title("Coverage vs Sparsity (Class Averages)")
plt.legend()
plt.grid()
plt.savefig("results/Tables/ClassCoveragePlot.png", dpi = 500) # Save Plot
