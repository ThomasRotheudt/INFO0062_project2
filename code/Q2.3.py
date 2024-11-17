import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Load the dataset
def load_wine_quality():
    """Loads and returns the (normalized) Wine Quality dataset from OpenML."""
    dataset = fetch_openml(data_id=287, parser='auto')

    x, y = dataset.data, dataset.target
    x, y = x.to_numpy(), y.to_numpy()

    # Normalization is important for ridge regression and k-NN.
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    # Shuffle the data
    x, y = shuffle(x, y, random_state=42)

    return x, y

# Step 1: Prepare the data
def prepare_learning_samples(X, y, num_samples=8, sample_size=500, test_size=0.2):
    """
    Split the dataset into train and test sets, and generate multiple learning samples.
    """
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Generate multiple learning samples from the training data
    learning_samples = []
    for _ in range(num_samples):
        idxs = np.random.choice(len(X_train), size=sample_size, replace=False)
        learning_samples.append((X_train[idxs], y_train[idxs]))
    
    return learning_samples, X_test, y_test

# Step 2: Train models and collect predictions
def train_and_predict(learning_samples, X_test, model_fn):
    """
    Train models on each learning sample and generate predictions on the test set.
    """
    predictions = []
    
    for X_train, y_train in learning_samples:
        # Train the model
        model = model_fn()
        model.fit(X_train, y_train)
        
        # Predict on the test set
        preds = model.predict(X_test)
        predictions.append(preds)
    
    return np.array(predictions)

# Step 3: Compute metrics (Total Error, Variance, Bias + Residual Error)
def compute_metrics(predictions, true_values):
    """
    Compute Total Error, Variance, and Bias + Residual Error.
    """
    # Number of models (M)
    M = predictions.shape[0]
    
    # Mean prediction across models
    mean_prediction = np.mean(predictions, axis=0)
    
    # Total Error
    total_error = np.mean((predictions - true_values) ** 2)
    
    # Variance
    variance = np.mean(np.mean((predictions - mean_prediction) ** 2, axis=0))
    
    # Bias + Residual Error
    bias_residual_error = total_error - variance
    
    return total_error, variance, bias_residual_error

# Plot the results
def plot_results(results):
    """
    Plot the total error, variance, and bias + residual error for different models.
    """
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

# Main program
if __name__ == "__main__":
    # Load the Wine Quality dataset
    X, y = load_wine_quality()
    
    # Prepare the data
    learning_samples, X_test, y_test = prepare_learning_samples(X, y, num_samples=10, sample_size=250, test_size=0.2)
    
    # Evaluate different models
    results = {}
    
    # k-NN
    knn_model_fn = lambda: KNeighborsRegressor(n_neighbors=5)
    knn_predictions = train_and_predict(learning_samples, X_test, knn_model_fn)
    total_error, variance, bias_residual_error = compute_metrics(knn_predictions, y_test)
    results["k-NN"] = {
        "total_error": total_error,
        "variance": variance,
        "bias_residual_error": bias_residual_error
    }
    
    # Lasso
    lasso_model_fn = lambda: Lasso(alpha=0.5**4)
    lasso_predictions = train_and_predict(learning_samples, X_test, lasso_model_fn)
    total_error, variance, bias_residual_error = compute_metrics(lasso_predictions, y_test)
    results["Lasso"] = {
        "total_error": total_error,
        "variance": variance,
        "bias_residual_error": bias_residual_error
    }
    
    # Decision Tree
    tree_model_fn = lambda: DecisionTreeRegressor(max_depth=5)
    tree_predictions = train_and_predict(learning_samples, X_test, tree_model_fn)
    total_error, variance, bias_residual_error = compute_metrics(tree_predictions, y_test)
    results["Decision Tree"] = {
        "total_error": total_error,
        "variance": variance,
        "bias_residual_error": bias_residual_error
    }
    
    # Plot the results
    plot_results(results)