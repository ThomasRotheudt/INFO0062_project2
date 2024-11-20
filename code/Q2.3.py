import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor as RegressionTree
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from data import load_wine_quality

def prepare_learning_samples(X, y, num_samples=80, sample_size=250, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    learning_samples = []
    for _ in range(num_samples):
        idxs = np.random.choice(len(X_train), size=sample_size, replace=False)
        learning_samples.append((X_train[idxs], y_train[idxs]))
    
    return learning_samples, X_test, y_test

def train_and_predict(learning_samples, X_test, model_fn):
    """
    Train models on each learning sample and generate predictions on the test set.
    """
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

# Plot the results
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

if __name__ == "__main__":
    X, y = load_wine_quality()
    
    learning_samples, X_test, y_test = prepare_learning_samples(X, y, num_samples=80, sample_size=250, test_size=0.2)
    
    results = {}

    knn_params = range(1, 21)
    knn_results = {"params": [], "total_errors": [], "variances": [], "biases": []}
    for k in knn_params:
        knn_model_fn = lambda: KNeighborsRegressor(n_neighbors=k)
        knn_predictions = train_and_predict(learning_samples, X_test, knn_model_fn)
        total_error, variance, bias_residual_error = compute_metrics(knn_predictions, y_test)
        knn_results["params"].append(k)
        knn_results["total_errors"].append(total_error)
        knn_results["variances"].append(variance)
        knn_results["biases"].append(bias_residual_error)
    results["k-NN"] = knn_results
    
    lasso_params = np.logspace(-6, 0, 20)  
    lasso_results = {"params": [], "total_errors": [], "variances": [], "biases": []}
    for alpha in lasso_params:
        lasso_model_fn = lambda: Lasso(alpha=alpha)
        lasso_predictions = train_and_predict(learning_samples, X_test, lasso_model_fn)
        total_error, variance, bias_residual_error = compute_metrics(lasso_predictions, y_test)
        lasso_results["params"].append(alpha)
        lasso_results["total_errors"].append(total_error)
        lasso_results["variances"].append(variance)
        lasso_results["biases"].append(bias_residual_error)
    results["Lasso"] = lasso_results
    
    tree_params = range(1, 21)
    tree_results = {"params": [], "total_errors": [], "variances": [], "biases": []}
    for max_depth in tree_params:
        tree_model_fn = lambda: RegressionTree(max_depth=max_depth)
        tree_predictions = train_and_predict(learning_samples, X_test, tree_model_fn)
        total_error, variance, bias_residual_error = compute_metrics(tree_predictions, y_test)
        tree_results["params"].append(max_depth)
        tree_results["total_errors"].append(total_error)
        tree_results["variances"].append(variance)
        tree_results["biases"].append(bias_residual_error)
    results["Regression Tree"] = tree_results
    
    plot_evolution(results)