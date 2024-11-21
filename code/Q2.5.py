import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor as RegressionTree
from data import load_wine_quality
from utils import *
from constants import *
from plots import *

X, y = load_wine_quality()

learning_samples, x_test, y_test = prepare_learning_samples(X, y, nb_samples=50, sample_size=250, test_size=0.2)

results_per_depth = {}
if CHANGE_DEPTH:
    for depth in TREE_PARAMS_Q2_5:
        results_per_depth[depth] = {}
else:
    results_per_depth[BASE_DEPTH] = {}

for key in results_per_depth:
    print(f"Starting depth {key}")
    # Bagging
    if BAGGING:
        bagging_results = {"n_estimators": [], "total_errors": [], "variances": [], "biases": []}
        for n_estimators in NB_ESTIMATOR:
            bagging_model = BaggingRegressor(estimator=RegressionTree(max_depth=key, random_state=SEED), n_estimators=n_estimators, random_state=SEED)
            bagging_predictions = train_and_predict(learning_samples, x_test, bagging_model)
            total_error, variance, bias_residual_error = compute_metrics(bagging_predictions, y_test)
            bagging_results["n_estimators"].append(n_estimators)
            bagging_results["total_errors"].append(total_error)
            bagging_results["variances"].append(variance)
            bagging_results["biases"].append(bias_residual_error)
        results_per_depth[key]["Bagging"] = bagging_results

    # Boosting
    if BOOSTING:
        boosting_results = {"n_estimators": [], "total_errors": [], "variances": [], "biases": []}
        for n_estimators in NB_ESTIMATOR:
            boosting_model = AdaBoostRegressor(estimator=RegressionTree(max_depth=key, random_state=SEED), n_estimators=n_estimators, random_state=SEED)
            boosting_predictions = train_and_predict(learning_samples, x_test, boosting_model)
            total_error, variance, bias_residual_error = compute_metrics(boosting_predictions, y_test)
            boosting_results["n_estimators"].append(n_estimators)
            boosting_results["total_errors"].append(total_error)
            boosting_results["variances"].append(variance)
            boosting_results["biases"].append(bias_residual_error)
        results_per_depth[key]["Boosting"] = boosting_results

if not CHANGE_DEPTH:
    plot_ensemble_results(results_per_depth[BASE_DEPTH])
else:
    plot_per_element(results_per_depth)
    pass
