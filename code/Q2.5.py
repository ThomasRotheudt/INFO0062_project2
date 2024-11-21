import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor as RegressionTree
from data import load_wine_quality
from utils import *
from constants import *

X, y = load_wine_quality()

learning_samples, x_test, y_test = prepare_learning_samples(X, y, nb_samples=50, sample_size=250, test_size=0.2)

results = {}

# Bagging
bagging_results = {"n_estimators": [], "total_errors": [], "variances": [], "biases": []}
for n_estimators in range(1, 21):
    bagging_model = BaggingRegressor(estimator=RegressionTree(max_depth=5, random_state=SEED), n_estimators=n_estimators, random_state=SEED)
    bagging_predictions = train_and_predict(learning_samples, x_test, bagging_model)
    total_error, variance, bias_residual_error = compute_metrics(bagging_predictions, y_test)
    bagging_results["n_estimators"].append(n_estimators)
    bagging_results["total_errors"].append(total_error)
    bagging_results["variances"].append(variance)
    bagging_results["biases"].append(bias_residual_error)
results["Bagging"] = bagging_results

# Boosting
boosting_results = {"n_estimators": [], "total_errors": [], "variances": [], "biases": []}
for n_estimators in range(1, 21):
    boosting_model = AdaBoostRegressor(estimator=RegressionTree(max_depth=5, random_state=SEED), n_estimators=n_estimators, random_state=SEED)
    boosting_predictions = train_and_predict(learning_samples, x_test, boosting_model)
    total_error, variance, bias_residual_error = compute_metrics(boosting_predictions, y_test)
    boosting_results["n_estimators"].append(n_estimators)
    boosting_results["total_errors"].append(total_error)
    boosting_results["variances"].append(variance)
    boosting_results["biases"].append(bias_residual_error)
results["Boosting"] = boosting_results

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

plot_ensemble_results(results)
