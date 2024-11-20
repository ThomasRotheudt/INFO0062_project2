import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor as RegressionTree
from sklearn.linear_model import Lasso
from data import load_wine_quality

from utils import *
from plots import *
from constants import *

X, y = load_wine_quality()
sample_sizes = SAMPLE_SIZES
results = {}
for sample_size in sample_sizes:
    learning_samples, x_test, y_test = prepare_learning_samples(X, y, nb_samples=50, sample_size=sample_size, test_size=0.2)
    results[sample_size] = {}
    k = 5
    knn_model_fn = lambda: KNeighborsRegressor(n_neighbors=k)
    knn_predictions = train_and_predict(learning_samples, x_test, knn_model_fn)
    total_error, variance, bias_residual_error = compute_metrics(knn_predictions, y_test)
    results[sample_size]["k-NN"] = {
        "total_error": total_error,
        "variance": variance,
        "bias_residual_error": bias_residual_error
    }
    alpha = 0.01
    lasso_model_fn = lambda: Lasso(alpha=alpha)
    lasso_predictions = train_and_predict(learning_samples, x_test, lasso_model_fn)
    total_error, variance, bias_residual_error = compute_metrics(lasso_predictions, y_test)
    results[sample_size]["Lasso"] = {
        "total_error": total_error,
        "variance": variance,
        "bias_residual_error": bias_residual_error
    }
    max_depth = 5
    tree_model_fn_fixed = lambda: RegressionTree(max_depth=max_depth)
    tree_predictions_fixed = train_and_predict(learning_samples, x_test, tree_model_fn_fixed)
    total_error_fixed, variance_fixed, bias_residual_error_fixed = compute_metrics(tree_predictions_fixed, y_test)
    tree_model_fn_full = lambda: RegressionTree()
    tree_predictions_full = train_and_predict(learning_samples, x_test, tree_model_fn_full)
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