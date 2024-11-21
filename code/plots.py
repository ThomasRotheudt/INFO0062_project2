import numpy as np
import matplotlib.pyplot as plt

from constants import *

def plot_learning_sample_size_impact(results):
    sample_sizes = list(results.keys())
    models = list(results[sample_sizes[0]].keys())
    for model in models:
        total_errors = [results[size][model]["total_error"] for size in sample_sizes]
        variances = [results[size][model]["variance"] for size in sample_sizes]
        biases = [results[size][model]["bias_residual_error"] for size in sample_sizes]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sample_sizes, total_errors, label="Total Error", linewidth=2, marker='o')
        ax.plot(sample_sizes, variances, label="Variance", linewidth=2, marker='s')
        ax.plot(sample_sizes, biases, label="Bias + Residual Error", linewidth=2, marker='^')
        ax.set_title(f"{model}")
        ax.set_xlabel("Learning Sample Size")
        ax.set_ylabel("Error")
        ax.legend()
        plt.tight_layout()
        plt.show()

def plot_results(results):
    models = list(results.keys())
    total_errors = [results[model]["total_error"] for model in models]
    variances = [results[model]["variance"] for model in models]
    biases = [results[model]["bias_residual_error"] for model in models]

    # Create a bar chart
    x = np.arange(len(models))
    width = 0.25

    plt.bar(x - width, total_errors, width, label="Total Error")
    plt.bar(x, variances, width, label="Variance")
    plt.bar(x + width, biases, width, label="Bias + Residual Error")

    plt.xlabel("Models")
    plt.ylabel("Error")
    plt.title("Comparison of Errors for Different Models")
    plt.xticks(x, models)
    plt.legend()
    plt.show()

def plot_evolution(results):
    param_names = {
        "k-NN": "Number of Neighbors (k)",
        "Lasso": "Regularization Parameter (λ)",
        "Regression Tree": "Max Depth"
    }

    for model in results:
        params = results[model]["params"]
        total_errors = results[model]["total_errors"]
        variances = results[model]["variances"]
        biases = results[model]["biases"]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(params, total_errors, label="Total Error", linewidth=2, marker='o')
        ax.plot(params, variances, label="Variance", linewidth=2, marker='s')
        ax.plot(params, biases, label="Bias + Residual Error", linewidth=2, marker='^')

        ax.set_title(f"{model}")
        ax.set_xlabel(param_names[model])
        ax.set_ylabel("Error")
        ax.legend()

        plt.tight_layout()
        plt.show()

def plot_ensemble_results(results):
    for method in results:
        n_estimators = results[method]["n_estimators"]
        total_errors = results[method]["total_errors"]
        variances = results[method]["variances"]
        biases = results[method]["biases"]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(n_estimators, total_errors, label="Total Error", linewidth=2, marker='o')
        ax.plot(n_estimators, variances, label="Variance", linewidth=2, marker='s')
        ax.plot(n_estimators, biases, label="Bias + Residual Error", linewidth=2, marker='^')

        ax.set_title(f"{method}")
        ax.set_xlabel("Number of Estimators")
        ax.set_ylabel("Error")
        ax.legend()

        plt.tight_layout()
        plt.show()

def plot_per_element(results_per_depth):
    for method in results_per_depth[TREE_PARAMS_Q2_5[0]]:
        for element in ["total_errors", "variances", "biases"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for depth in TREE_PARAMS_Q2_5:
                ax.plot(results_per_depth[depth][method]["n_estimators"], results_per_depth[depth][method][element], label=f"{element} with depth={depth}", linewidth=2, marker='o')
            
            ax.set_title(f"{element} of {method}")
            ax.set_xlabel("Number of Estimators")
            ax.set_ylabel("Value")
            ax.legend()

            plt.tight_layout()
            plt.show()
