import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor as RegressionTree
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from data import load_wine_quality

def prepare_learning_samples(X, y, num_samples, sample_size, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    learning_samples = []
    for _ in range(num_samples):
        idxs = np.random.choice(len(X_train), size=sample_size, replace=False)
        learning_samples.append((X_train[idxs], y_train[idxs]))
    return learning_samples, X_test, y_test

def train_and_predict(learning_samples, X_test, model_fn):
    predictions = []
    for X_train, y_train in learning_samples:
        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions.append(preds)
    return np.array(predictions)

def compute_metrics(predictions, true_values):
    total_error = np.mean((predictions - true_values) ** 2)
    mean_prediction = np.mean(predictions, axis=0)
    variance = np.var(predictions, axis=0)
    bias_squared = np.mean((mean_prediction - true_values) ** 2)
    
    return total_error, np.mean(variance), bias_squared

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

if __name__ == "__main__":
    X, y = load_wine_quality()
    sample_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 700, 1000, 1300, 1500, 1750, 2000]
    results = {}
    for sample_size in sample_sizes:
        learning_samples, X_test, y_test = prepare_learning_samples(X, y, num_samples=50, sample_size=sample_size, test_size=0.2)
        results[sample_size] = {}
        k = 5
        knn_model_fn = lambda: KNeighborsRegressor(n_neighbors=k)
        knn_predictions = train_and_predict(learning_samples, X_test, knn_model_fn)
        total_error, variance, bias_residual_error = compute_metrics(knn_predictions, y_test)
        results[sample_size]["k-NN"] = {
            "total_error": total_error,
            "variance": variance,
            "bias_residual_error": bias_residual_error
        }
        alpha = 0.01
        lasso_model_fn = lambda: Lasso(alpha=alpha)
        lasso_predictions = train_and_predict(learning_samples, X_test, lasso_model_fn)
        total_error, variance, bias_residual_error = compute_metrics(lasso_predictions, y_test)
        results[sample_size]["Lasso"] = {
            "total_error": total_error,
            "variance": variance,
            "bias_residual_error": bias_residual_error
        }
        max_depth = 5
        tree_model_fn_fixed = lambda: RegressionTree(max_depth=max_depth)
        tree_predictions_fixed = train_and_predict(learning_samples, X_test, tree_model_fn_fixed)
        total_error_fixed, variance_fixed, bias_residual_error_fixed = compute_metrics(tree_predictions_fixed, y_test)
        tree_model_fn_full = lambda: RegressionTree()
        tree_predictions_full = train_and_predict(learning_samples, X_test, tree_model_fn_full)
        total_error_full, variance_full, bias_residual_error_full = compute_metrics(tree_predictions_full, y_test)
        results[sample_size]["Regression Tree (Fixed Depth)"] = {
            "total_error": total_error_fixed,
            "variance": variance_fixed,
            "bias_residual_error": bias_residual_error_fixed
        }
        results[sample_size]["Regression Tree (Full)"] = {
            "total_error": total_error_full,
            "variance": variance_full,
            "bias_residual_error": bias_residual_error_full
        }
    plot_learning_sample_size_impact(results)