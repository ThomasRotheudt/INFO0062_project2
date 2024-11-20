import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor as RegressionTree
from sklearn.linear_model import Lasso
from data import load_wine_quality

from utils import *
from plots import *

# Load the dataset
X, y = load_wine_quality()
# Split the dataset
learning_samples, x_test, y_test = prepare_learning_samples(X, y, nb_samples=80, sample_size=250, test_size=0.2)

results = {}

if KNN:
    knn_params = KNN_PARAMS
    knn_results = {"params": [], "total_errors": [], "variances": [], "biases": []}
    for k in knn_params:
        knn_model = KNeighborsRegressor(n_neighbors=k)
        knn_predictions = train_and_predict(learning_samples, x_test, knn_model)
        total_error, variance, bias_residual_error = compute_metrics(knn_predictions, y_test)
        knn_results["params"].append(k)
        knn_results["total_errors"].append(total_error)
        knn_results["variances"].append(variance)
        knn_results["biases"].append(bias_residual_error)
    results["k-NN"] = knn_results

if LASSO:
    lasso_params = LASSO_PARAMS
    lasso_results = {"params": [], "total_errors": [], "variances": [], "biases": []}
    for alpha in lasso_params:
        lasso_model = Lasso(alpha=alpha, random_state=SEED)
        lasso_predictions = train_and_predict(learning_samples, x_test, lasso_model)
        total_error, variance, bias_residual_error = compute_metrics(lasso_predictions, y_test)
        lasso_results["params"].append(alpha)
        lasso_results["total_errors"].append(total_error)
        lasso_results["variances"].append(variance)
        lasso_results["biases"].append(bias_residual_error)
    results["Lasso"] = lasso_results

if TREE:
    tree_params = TREE_PARAMS
    tree_results = {"params": [], "total_errors": [], "variances": [], "biases": []}
    for max_depth in tree_params:
        tree_model = RegressionTree(max_depth=max_depth, random_state=SEED)
        tree_predictions = train_and_predict(learning_samples, x_test, tree_model)
        total_error, variance, bias_residual_error = compute_metrics(tree_predictions, y_test)
        tree_results["params"].append(max_depth)
        tree_results["total_errors"].append(total_error)
        tree_results["variances"].append(variance)
        tree_results["biases"].append(bias_residual_error)
    results["Regression Tree"] = tree_results

plot_evolution(results)